#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_mem.h"
#include <string.h>

/*
 * Dropout (训练态)  cfg.param = drop prob (0..1)
 *                   cfg.activation = 0 train / 1 inference
 * 推理时直接 passthrough.
 */
typedef struct {
    bnn_layer_t base;
    float p;
    int   train;
    unsigned int rng;
    unsigned char *mask;  /* 0/1 bytes, size = numel */
    size_t mask_cap;
} drop_t;

static unsigned int xrand(unsigned int *s) { *s = (*s) * 1664525u + 1013904223u; return *s; }

static bnn_layer_t *dp_create(const bnn_layer_cfg_t *cfg);
static void         dp_destroy(bnn_layer_t *self);
static void         dp_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*dp_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         dp_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                const float *dy, float **dx);
static bnn_layer_param_ref_t *dp_params(bnn_layer_t *self) { (void)self; return NULL; }

static const bnn_layer_vtbl_t dp_vtbl = {
    .create = dp_create, .destroy = dp_destroy, .infer_shape = dp_infer,
    .forward = dp_forward, .backward = dp_backward, .params = dp_params,
};
BNN_REGISTER_LAYER(dropout, &dp_vtbl)

static bnn_layer_t *dp_create(const bnn_layer_cfg_t *cfg) {
    drop_t *d = (drop_t *)bnn_calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->base.vtbl = &dp_vtbl;
    d->p   = cfg ? cfg->param : 0.5f;
    d->train = cfg ? (cfg->activation == 0 ? 1 : 0) : 1;
    d->rng = 0xC0FFEEu;
    return &d->base;
}
static void dp_destroy(bnn_layer_t *self) {
    if (!self) return;
    drop_t *d = (drop_t *)self;
    if (d->mask) bnn_free(d->mask);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(d);
}
static void dp_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self; for (int i=0;i<in;++i) os[i]=is[i]; *on=in;
}
static bnn_tensor_t *dp_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    drop_t *d = (drop_t *)self;
    bnn_tensor_t *x = inputs[0];
    if (self->cached_output && self->cached_output->numel != x->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, x->ndim, x->shape);
        if (!self->cached_output) return NULL;
    }
    const float *xp = (const float *)x->data;
    float *yp = (float *)self->cached_output->data;
    if (!d->train || d->p <= 0.f) {
        memcpy(yp, xp, x->numel * sizeof(float));
        return self->cached_output;
    }
    if (d->mask_cap < x->numel) {
        if (d->mask) bnn_free(d->mask);
        d->mask = (unsigned char *)bnn_malloc(x->numel);
        d->mask_cap = x->numel;
    }
    float scale = 1.f / (1.f - d->p);
    for (size_t i = 0; i < x->numel; ++i) {
        float u = (xrand(&d->rng) & 0xFFFFFF) / (float)0x1000000;
        unsigned char m = u >= d->p ? 1 : 0;
        d->mask[i] = m;
        yp[i] = m ? xp[i] * scale : 0.f;
    }
    return self->cached_output;
}
static void dp_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                        const float *dy, float **dx) {
    (void)n_in;
    drop_t *d = (drop_t *)self;
    if (!dx || !dx[0]) return;
    size_t n = inputs[0]->numel;
    if (!d->train || d->p <= 0.f) { memcpy(dx[0], dy, n * sizeof(float)); return; }
    float scale = 1.f / (1.f - d->p);
    for (size_t i = 0; i < n; ++i) dx[0][i] = d->mask[i] ? dy[i] * scale : 0.f;
}
