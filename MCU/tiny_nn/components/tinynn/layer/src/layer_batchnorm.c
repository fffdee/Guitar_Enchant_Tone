#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

/*
 * BatchNorm1d:
 *  输入: [B, F]
 *  训练: y = gamma * (x - mu)/sqrt(var+eps) + beta
 *        mu, var per-feature; running stats moving avg
 *  推理: 使用 running_mean / running_var
 *
 *  cfg.in_features = F
 *  cfg.param       = momentum (默认 0.1)
 *  cfg.activation  =  0 训练态, 1 推理态  (作为模式开关复用字段)
 */
typedef struct {
    bnn_layer_t base;
    int F;
    float eps;
    float momentum;
    int   train_mode;     /* 1=train */
    float *gamma, *beta;
    float *dgamma, *dbeta;
    float *run_mean, *run_var;
    /* 缓存供反向 */
    float *xhat;
    float *istd;
    float *mu;
    int    B_cached;
    bnn_layer_param_ref_t pref_g, pref_b;
} bn_t;

static bnn_layer_t *bn_create(const bnn_layer_cfg_t *cfg);
static void         bn_destroy(bnn_layer_t *self);
static void         bn_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*bn_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         bn_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                const float *dy, float **dx);
static bnn_layer_param_ref_t *bn_params(bnn_layer_t *self);

static const bnn_layer_vtbl_t bn_vtbl = {
    .create = bn_create, .destroy = bn_destroy, .infer_shape = bn_infer,
    .forward = bn_forward, .backward = bn_backward, .params = bn_params,
};

BNN_REGISTER_LAYER(batchnorm1d, &bn_vtbl)

static bnn_layer_t *bn_create(const bnn_layer_cfg_t *cfg) {
    if (!cfg || cfg->in_features <= 0) return NULL;
    bn_t *b = (bn_t *)bnn_calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->base.vtbl = &bn_vtbl;
    b->F = cfg->in_features;
    b->eps = 1e-5f;
    b->momentum = cfg->param > 0.f ? cfg->param : 0.1f;
    b->train_mode = cfg->activation == 0 ? 1 : 0; /* default train */
    b->gamma   = (float *)bnn_calloc(b->F, sizeof(float));
    b->beta    = (float *)bnn_calloc(b->F, sizeof(float));
    b->dgamma  = (float *)bnn_calloc(b->F, sizeof(float));
    b->dbeta   = (float *)bnn_calloc(b->F, sizeof(float));
    b->run_mean= (float *)bnn_calloc(b->F, sizeof(float));
    b->run_var = (float *)bnn_calloc(b->F, sizeof(float));
    if (!b->gamma || !b->beta || !b->run_mean || !b->run_var) { bn_destroy(&b->base); return NULL; }
    for (int i = 0; i < b->F; ++i) { b->gamma[i] = 1.f; b->run_var[i] = 1.f; }
    b->pref_g.data = b->gamma; b->pref_g.grad = b->dgamma; b->pref_g.numel = (size_t)b->F;
    b->pref_g.next = &b->pref_b;
    b->pref_b.data = b->beta;  b->pref_b.grad = b->dbeta;  b->pref_b.numel = (size_t)b->F;
    b->pref_b.next = NULL;
    return &b->base;
}
static void bn_destroy(bnn_layer_t *self) {
    if (!self) return;
    bn_t *b = (bn_t *)self;
    if (b->gamma) bnn_free(b->gamma);
    if (b->beta)  bnn_free(b->beta);
    if (b->dgamma)bnn_free(b->dgamma);
    if (b->dbeta) bnn_free(b->dbeta);
    if (b->run_mean) bnn_free(b->run_mean);
    if (b->run_var)  bnn_free(b->run_var);
    if (b->xhat) bnn_free(b->xhat);
    if (b->istd) bnn_free(b->istd);
    if (b->mu)   bnn_free(b->mu);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(b);
}
static void bn_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self; for (int i=0;i<in;++i) os[i]=is[i]; *on=in;
}
static bnn_tensor_t *bn_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    bn_t *b = (bn_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0], F = b->F;
    if (self->cached_output && self->cached_output->numel != x->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, x->ndim, x->shape);
        if (!self->cached_output) return NULL;
    }
    if (b->B_cached != B) {
        if (b->xhat) bnn_free(b->xhat);
        if (b->istd) bnn_free(b->istd);
        if (b->mu)   bnn_free(b->mu);
        b->xhat = (float *)bnn_calloc((size_t)B * F, sizeof(float));
        b->istd = (float *)bnn_calloc(F, sizeof(float));
        b->mu   = (float *)bnn_calloc(F, sizeof(float));
        b->B_cached = B;
    }
    const float *xp = (const float *)x->data;
    float *yp = (float *)self->cached_output->data;
    if (b->train_mode) {
        for (int f = 0; f < F; ++f) {
            float s = 0.f;
            for (int i = 0; i < B; ++i) s += xp[i * F + f];
            float mu = s / B;
            float vs = 0.f;
            for (int i = 0; i < B; ++i) { float d = xp[i*F+f] - mu; vs += d*d; }
            float var = vs / B;
            float istd = 1.f / sqrtf(var + b->eps);
            b->mu[f] = mu; b->istd[f] = istd;
            b->run_mean[f] = (1.f - b->momentum) * b->run_mean[f] + b->momentum * mu;
            b->run_var[f]  = (1.f - b->momentum) * b->run_var[f]  + b->momentum * var;
            float g = b->gamma[f], bt = b->beta[f];
            for (int i = 0; i < B; ++i) {
                float xh = (xp[i*F+f] - mu) * istd;
                b->xhat[i*F+f] = xh;
                yp[i*F+f] = g * xh + bt;
            }
        }
    } else {
        for (int f = 0; f < F; ++f) {
            float istd = 1.f / sqrtf(b->run_var[f] + b->eps);
            float mu = b->run_mean[f];
            float g = b->gamma[f], bt = b->beta[f];
            for (int i = 0; i < B; ++i)
                yp[i*F+f] = g * (xp[i*F+f] - mu) * istd + bt;
        }
    }
    return self->cached_output;
}
static void bn_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                        const float *dy, float **dx) {
    (void)n_in;
    bn_t *b = (bn_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0], F = b->F;
    /* 标准 BN 反向 */
    for (int f = 0; f < F; ++f) {
        float dgamma = 0.f, dbeta = 0.f;
        float istd = b->istd[f], g = b->gamma[f];
        for (int i = 0; i < B; ++i) {
            float dyv = dy[i*F+f];
            dgamma += dyv * b->xhat[i*F+f];
            dbeta  += dyv;
        }
        b->dgamma[f] += dgamma;
        b->dbeta[f]  += dbeta;
        if (dx && dx[0]) {
            /* dx = (g * istd / B) * (B*dy - sum(dy) - xhat * sum(dy*xhat)) */
            float coef = g * istd / B;
            for (int i = 0; i < B; ++i) {
                dx[0][i*F+f] = coef * (B * dy[i*F+f] - dbeta - b->xhat[i*F+f] * dgamma);
            }
        }
    }
}
static bnn_layer_param_ref_t *bn_params(bnn_layer_t *self) { return &((bn_t *)self)->pref_g; }
