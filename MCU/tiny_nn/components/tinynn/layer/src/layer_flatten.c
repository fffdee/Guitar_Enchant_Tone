#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_mem.h"
#include <string.h>

/* Flatten: 保留 batch 维, 其余 flatten 成一维. */
typedef struct { bnn_layer_t base; int in_shape[BNN_TENSOR_MAX_DIM]; int in_ndim; } fl_t;

static bnn_layer_t *fl_create(const bnn_layer_cfg_t *cfg);
static void         fl_destroy(bnn_layer_t *self);
static void         fl_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*fl_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         fl_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                const float *dy, float **dx);
static bnn_layer_param_ref_t *fl_params(bnn_layer_t *self) { (void)self; return NULL; }

static const bnn_layer_vtbl_t fl_vtbl = {
    .create = fl_create, .destroy = fl_destroy, .infer_shape = fl_infer,
    .forward = fl_forward, .backward = fl_backward, .params = fl_params,
};
BNN_REGISTER_LAYER(flatten, &fl_vtbl)

static bnn_layer_t *fl_create(const bnn_layer_cfg_t *cfg) {
    (void)cfg;
    fl_t *f = (fl_t *)bnn_calloc(1, sizeof(*f));
    if (f) f->base.vtbl = &fl_vtbl;
    return &f->base;
}
static void fl_destroy(bnn_layer_t *self) {
    if (!self) return;
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(self);
}
static void fl_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self;
    int n = 1; for (int i = 1; i < in; ++i) n *= is[i];
    os[0] = is[0]; os[1] = n; *on = 2;
}
static bnn_tensor_t *fl_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    fl_t *f = (fl_t *)self;
    bnn_tensor_t *x = inputs[0];
    f->in_ndim = x->ndim;
    for (int i = 0; i < x->ndim; ++i) f->in_shape[i] = x->shape[i];
    int B = x->shape[0];
    int n = (int)(x->numel / (size_t)B);
    int os[2] = { B, n };
    if (self->cached_output && self->cached_output->numel != x->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 2, os);
        if (!self->cached_output) return NULL;
    } else {
        self->cached_output->shape[0] = B;
        self->cached_output->shape[1] = n;
    }
    memcpy(self->cached_output->data, x->data, x->numel * sizeof(float));
    return self->cached_output;
}
static void fl_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                        const float *dy, float **dx) {
    (void)self; (void)n_in;
    if (dx && dx[0]) memcpy(dx[0], dy, inputs[0]->numel * sizeof(float));
}
