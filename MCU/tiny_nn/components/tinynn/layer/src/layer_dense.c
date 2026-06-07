#include "bnn_layer/bnn_layer.h"
#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

/*
 * Dense (全连接) 层
 *  输入  X : [B, in_features]
 *  权重  W : [in_features, out_features]
 *  偏置  b : [out_features]
 *  输出  Y = X @ W + b : [B, out_features]
 */
typedef struct {
    bnn_layer_t base;
    int in_f, out_f;
    float *W;       /* in_f * out_f */
    float *b;       /* out_f */
    float *dW;
    float *db;
    bnn_layer_param_ref_t pref_w, pref_b;
} dense_t;

static void he_init(float *p, int fan_in, int n) {
    /* 简单线性同余, 避免依赖 rand 状态; 嵌入式可重复 */
    static unsigned int s = 0x12345678;
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = ((s >> 8) & 0xFFFFFF) / (float)0x1000000;  /* [0,1) */
        s = s * 1664525u + 1013904223u;
        float v = ((s >> 8) & 0xFFFFFF) / (float)0x1000000;
        if (v < 1e-7f) v = 1e-7f;
        /* Box-Muller */
        float g = sqrtf(-2.f * logf(v)) * cosf(6.2831853f * u);
        p[i] = g * scale;
    }
}

static bnn_layer_t *dense_create(const bnn_layer_cfg_t *cfg);
static void         dense_destroy(bnn_layer_t *self);
static void         dense_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*dense_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         dense_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                   const float *dy, float **dx);
static bnn_layer_param_ref_t *dense_params(bnn_layer_t *self);

static const bnn_layer_vtbl_t dense_vtbl = {
    .create      = dense_create,
    .destroy     = dense_destroy,
    .infer_shape = dense_infer,
    .forward     = dense_forward,
    .backward    = dense_backward,
    .params      = dense_params,
};

BNN_REGISTER_LAYER(dense, &dense_vtbl)

static bnn_layer_t *dense_create(const bnn_layer_cfg_t *cfg) {
    if (!cfg || cfg->in_features <= 0 || cfg->out_features <= 0) {
        BNN_LOGE("dense: invalid cfg"); return NULL;
    }
    dense_t *d = (dense_t *)bnn_calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->base.vtbl = &dense_vtbl;
    d->in_f = cfg->in_features;
    d->out_f = cfg->out_features;
    int wn = d->in_f * d->out_f;
    d->W  = (float *)bnn_calloc(wn, sizeof(float));
    d->b  = (float *)bnn_calloc(d->out_f, sizeof(float));
    d->dW = (float *)bnn_calloc(wn, sizeof(float));
    d->db = (float *)bnn_calloc(d->out_f, sizeof(float));
    if (!d->W || !d->b || !d->dW || !d->db) { dense_destroy(&d->base); return NULL; }
    he_init(d->W, d->in_f, wn);
    /* 参数引用 */
    d->pref_w.data = d->W; d->pref_w.grad = d->dW; d->pref_w.numel = (size_t)wn;
    d->pref_w.next = &d->pref_b;
    d->pref_b.data = d->b; d->pref_b.grad = d->db; d->pref_b.numel = (size_t)d->out_f;
    d->pref_b.next = NULL;
    return &d->base;
}

static void dense_destroy(bnn_layer_t *self) {
    if (!self) return;
    dense_t *d = (dense_t *)self;
    if (d->W) bnn_free(d->W);
    if (d->b) bnn_free(d->b);
    if (d->dW) bnn_free(d->dW);
    if (d->db) bnn_free(d->db);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(d);
}

static void dense_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    dense_t *d = (dense_t *)self;
    int B = (in >= 2) ? is[0] : 1;
    os[0] = B; os[1] = d->out_f; *on = 2;
}

static bnn_tensor_t *dense_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    dense_t *d = (dense_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0];
    int shape[2] = { B, d->out_f };
    if (self->cached_output) {
        if (self->cached_output->shape[0] != B) {
            bnn_tensor_release(self->cached_output);
            self->cached_output = NULL;
        }
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 2, shape);
        if (!self->cached_output) return NULL;
    }
    self->cached_input = x; /* 仅缓存指针 (前向输出生命周期由 graph 持有) */
    BNN_OP()->gemm((const float *)x->data, d->W, d->b,
                   (float *)self->cached_output->data,
                   B, d->out_f, d->in_f, 0);
    return self->cached_output;
}

static void dense_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                           const float *dy, float **dx) {
    (void)n_in;
    dense_t *d = (dense_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0];
    const float *X = (const float *)x->data;

    /* dW[in_f, out_f] += X^T @ dY -> 用 gemm: (in_f x B) * (B x out_f) */
    /* 等价: for b,i,j  dW[i,j] += X[b,i]*dY[b,j] */
    for (int bi = 0; bi < B; ++bi) {
        const float *xr = X  + bi * d->in_f;
        const float *yr = dy + bi * d->out_f;
        for (int i = 0; i < d->in_f; ++i) {
            float a = xr[i];
            float *wr = d->dW + i * d->out_f;
            for (int j = 0; j < d->out_f; ++j) wr[j] += a * yr[j];
        }
        for (int j = 0; j < d->out_f; ++j) d->db[j] += yr[j];
    }

    /* dX = dY @ W^T : [B, in_f] */
    if (dx && dx[0]) {
        float *DX = dx[0];
        memset(DX, 0, sizeof(float) * (size_t)B * d->in_f);
        for (int bi = 0; bi < B; ++bi) {
            const float *yr = dy + bi * d->out_f;
            float *xr = DX + bi * d->in_f;
            for (int i = 0; i < d->in_f; ++i) {
                const float *wr = d->W + i * d->out_f;
                float s = 0.f;
                for (int j = 0; j < d->out_f; ++j) s += yr[j] * wr[j];
                xr[i] = s;
            }
        }
    }
}

static bnn_layer_param_ref_t *dense_params(bnn_layer_t *self) {
    return &((dense_t *)self)->pref_w;
}
