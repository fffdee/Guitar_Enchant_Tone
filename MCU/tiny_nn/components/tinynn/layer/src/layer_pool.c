#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <float.h>
#include <string.h>

/*
 * MaxPool2d / AvgPool2d  (NCHW)
 *  cfg.kernel, cfg.stride (default = kernel), cfg.padding
 *  cfg.activation: 0 = max, 1 = avg
 */
typedef struct {
    bnn_layer_t base;
    int K, S, P;
    int kind;         /* 0:max 1:avg */
    int Cin_cached, Hin_cached, Win_cached;
    int Ho_cached, Wo_cached;
    int *argidx;      /* max 用: 每输出元素记录输入索引, 大小 = numel(out) */
    int argidx_cap;
} pool_t;

static bnn_layer_t *pool_create(const bnn_layer_cfg_t *cfg);
static void         pool_destroy(bnn_layer_t *self);
static void         pool_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*pool_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         pool_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                  const float *dy, float **dx);
static bnn_layer_param_ref_t *pool_params(bnn_layer_t *self) { (void)self; return NULL; }

static const bnn_layer_vtbl_t pool_vtbl = {
    .create = pool_create, .destroy = pool_destroy, .infer_shape = pool_infer,
    .forward = pool_forward, .backward = pool_backward, .params = pool_params,
};
BNN_REGISTER_LAYER(pool2d, &pool_vtbl)

static bnn_layer_t *pool_create(const bnn_layer_cfg_t *cfg) {
    pool_t *p = (pool_t *)bnn_calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->base.vtbl = &pool_vtbl;
    p->K = cfg ? cfg->kernel  : 2;
    p->S = cfg && cfg->stride > 0 ? cfg->stride : p->K;
    p->P = cfg ? cfg->padding : 0;
    p->kind = cfg ? cfg->activation : 0;
    if (p->K <= 0) p->K = 2;
    return &p->base;
}
static void pool_destroy(bnn_layer_t *self) {
    if (!self) return;
    pool_t *p = (pool_t *)self;
    if (p->argidx) bnn_free(p->argidx);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(p);
}
static void pool_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    pool_t *p = (pool_t *)self; (void)in;
    int B = is[0], C = is[1], H = is[2], W = is[3];
    int Ho = (H + 2*p->P - p->K)/p->S + 1;
    int Wo = (W + 2*p->P - p->K)/p->S + 1;
    os[0]=B; os[1]=C; os[2]=Ho; os[3]=Wo; *on=4;
}
static bnn_tensor_t *pool_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    pool_t *p = (pool_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B=x->shape[0], C=x->shape[1], H=x->shape[2], W=x->shape[3];
    int Ho=(H+2*p->P-p->K)/p->S+1, Wo=(W+2*p->P-p->K)/p->S+1;
    int os[4] = { B, C, Ho, Wo };
    p->Cin_cached=C; p->Hin_cached=H; p->Win_cached=W; p->Ho_cached=Ho; p->Wo_cached=Wo;
    if (self->cached_output && (self->cached_output->shape[0]!=B ||
                                self->cached_output->shape[2]!=Ho ||
                                self->cached_output->shape[3]!=Wo)) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 4, os);
        if (!self->cached_output) return NULL;
    }
    int outn = B*C*Ho*Wo;
    if (p->kind == 0) {
        if (p->argidx_cap < outn) {
            if (p->argidx) bnn_free(p->argidx);
            p->argidx = (int *)bnn_malloc(sizeof(int)*outn);
            p->argidx_cap = outn;
        }
    }
    const float *X = (const float *)x->data;
    float *Y = (float *)self->cached_output->data;
    float inv = 1.f / (float)(p->K * p->K);
    int oi = 0;
    for (int b=0;b<B;++b)
    for (int c=0;c<C;++c)
    for (int oh=0;oh<Ho;++oh)
    for (int ow=0;ow<Wo;++ow, ++oi) {
        if (p->kind == 0) {
            float mv = -FLT_MAX; int midx = -1;
            for (int kh=0;kh<p->K;++kh)
            for (int kw=0;kw<p->K;++kw) {
                int ih = oh*p->S - p->P + kh, iw = ow*p->S - p->P + kw;
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                    int xi = ((b*C+c)*H+ih)*W+iw;
                    if (X[xi] > mv) { mv = X[xi]; midx = xi; }
                }
            }
            Y[oi] = mv == -FLT_MAX ? 0.f : mv;
            p->argidx[oi] = midx;
        } else {
            float s = 0.f;
            for (int kh=0;kh<p->K;++kh)
            for (int kw=0;kw<p->K;++kw) {
                int ih = oh*p->S - p->P + kh, iw = ow*p->S - p->P + kw;
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                    s += X[((b*C+c)*H+ih)*W+iw];
            }
            Y[oi] = s * inv;
        }
    }
    return self->cached_output;
}
static void pool_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                          const float *dy, float **dx) {
    (void)n_in;
    if (!dx || !dx[0]) return;
    pool_t *p = (pool_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B=x->shape[0], C=x->shape[1], H=p->Hin_cached, W=p->Win_cached;
    int Ho=p->Ho_cached, Wo=p->Wo_cached;
    memset(dx[0], 0, sizeof(float)*B*C*H*W);
    float inv = 1.f / (float)(p->K*p->K);
    int oi = 0;
    for (int b=0;b<B;++b)
    for (int c=0;c<C;++c)
    for (int oh=0;oh<Ho;++oh)
    for (int ow=0;ow<Wo;++ow, ++oi) {
        if (p->kind == 0) {
            int mi = p->argidx[oi];
            if (mi >= 0) dx[0][mi] += dy[oi];
        } else {
            float gv = dy[oi] * inv;
            for (int kh=0;kh<p->K;++kh)
            for (int kw=0;kw<p->K;++kw) {
                int ih = oh*p->S - p->P + kh, iw = ow*p->S - p->P + kw;
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                    dx[0][((b*C+c)*H+ih)*W+iw] += gv;
            }
        }
    }
}
