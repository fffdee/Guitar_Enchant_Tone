#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

/*
 * Residual Add: y = a + b, 两个输入需 shape 完全一致.
 * backward: dx[0] = dy, dx[1] = dy
 */
typedef struct { bnn_layer_t base; } res_t;

/* vtbl 由 bnn_layer_create 工厂统一回填, create 内无需设置 */
static bnn_layer_t *res_create(const bnn_layer_cfg_t *cfg) {
    (void)cfg;
    res_t *r = (res_t *)bnn_calloc(1, sizeof(*r));
    return r ? &r->base : NULL;
}
static void res_destroy(bnn_layer_t *self) {
    if (!self) return;
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(self);
}
static void res_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self;
    for (int i = 0; i < in; ++i) os[i] = is[i];
    *on = in;
}
static bnn_tensor_t *res_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    if (n_in < 2) { BNN_LOGE("residual needs 2 inputs"); return NULL; }
    bnn_tensor_t *a = inputs[0], *b = inputs[1];
    if (a->numel != b->numel) { BNN_LOGE("residual shape mismatch"); return NULL; }
    if (self->cached_output && self->cached_output->numel != a->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, a->ndim, a->shape);
        if (!self->cached_output) return NULL;
    }
    const float *pa = (const float *)a->data;
    const float *pb = (const float *)b->data;
    float *pc = (float *)self->cached_output->data;
    for (size_t i = 0; i < a->numel; ++i) pc[i] = pa[i] + pb[i];
    return self->cached_output;
}
static void res_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                         const float *dy, float **dx) {
    (void)self;
    bnn_tensor_t *a = inputs[0];
    size_t n = a->numel;
    if (n_in >= 1 && dx && dx[0]) memcpy(dx[0], dy, n * sizeof(float));
    if (n_in >= 2 && dx && dx[1]) memcpy(dx[1], dy, n * sizeof(float));
}
static bnn_layer_param_ref_t *res_params(bnn_layer_t *self) { (void)self; return NULL; }

static const bnn_layer_vtbl_t res_vtbl = {
    .create = res_create, .destroy = res_destroy, .infer_shape = res_infer,
    .forward = res_forward, .backward = res_backward, .params = res_params,
};

BNN_REGISTER_LAYER(residual, &res_vtbl)
