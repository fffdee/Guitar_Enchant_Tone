#include "bnn_layer/bnn_layer.h"
#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

/*
 * 激活层: 0:identity 1:relu 2:sigmoid 3:tanh 4:softplus
 *   输出缩放 scale = cfg.param (<=0 视为 1.0), 输出 = scale * act(x).
 *   用于谱掩码三头: sigmoid×Gmax / tanh×Δφmax / softplus×1.
 * 设计为单独的 layer, 便于在计算图中复用.
 */
typedef struct {
    bnn_layer_t base;
    int   kind;
    float scale;
} act_t;

static bnn_layer_t *act_create(const bnn_layer_cfg_t *cfg);
static void         act_destroy(bnn_layer_t *self);
static void         act_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*act_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         act_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                 const float *dy, float **dx);
static bnn_layer_param_ref_t *act_params(bnn_layer_t *self) { (void)self; return NULL; }

static const bnn_layer_vtbl_t act_vtbl = {
    .create = act_create, .destroy = act_destroy, .infer_shape = act_infer,
    .forward = act_forward, .backward = act_backward, .params = act_params,
};

BNN_REGISTER_LAYER(activation, &act_vtbl)

static bnn_layer_t *act_create(const bnn_layer_cfg_t *cfg) {
    act_t *a = (act_t *)bnn_calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->base.vtbl = &act_vtbl;
    a->kind = cfg ? cfg->activation : 1;
    a->scale = (cfg && cfg->param > 0.0f) ? cfg->param : 1.0f;
    return &a->base;
}

static float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}
static void act_destroy(bnn_layer_t *self) {
    if (!self) return;
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(self);
}
static void act_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self;
    for (int i = 0; i < in; ++i) os[i] = is[i];
    *on = in;
}
static bnn_tensor_t *act_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    act_t *a = (act_t *)self;
    bnn_tensor_t *x = inputs[0];
    if (self->cached_output && self->cached_output->numel != x->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, x->ndim, x->shape);
        if (!self->cached_output) return NULL;
    }
    self->cached_input = x;
    const float *xp = (const float *)x->data;
    float *yp = (float *)self->cached_output->data;
    size_t n = x->numel;
    switch (a->kind) {
        case 1: BNN_OP()->relu(xp, yp, n); break;
        case 2: BNN_OP()->sigmoid(xp, yp, n); break;
        case 3: BNN_OP()->tanh(xp, yp, n); break;
        case 4: for (size_t i = 0; i < n; ++i) yp[i] = softplus_f(xp[i]); break;
        default: memcpy(yp, xp, n * sizeof(float)); break;
    }
    if (a->scale != 1.0f) for (size_t i = 0; i < n; ++i) yp[i] *= a->scale;
    return self->cached_output;
}
static void act_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                         const float *dy, float **dx) {
    (void)n_in;
    if (!dx || !dx[0]) return;
    act_t *a = (act_t *)self;
    bnn_tensor_t *x = inputs[0];
    const float *xp = (const float *)x->data;
    const float *yp = (const float *)self->cached_output->data;
    float *dxp = dx[0];
    size_t n = x->numel;
    float s = a->scale, inv = (s != 0.0f) ? 1.0f / s : 1.0f;
    switch (a->kind) {
        case 1:
            BNN_OP()->relu_grad(xp, dy, dxp, n);
            if (s != 1.0f) for (size_t i = 0; i < n; ++i) dxp[i] *= s;
            break;
        case 2: for (size_t i = 0; i < n; ++i) { float b = yp[i] * inv; dxp[i] = dy[i] * s * b * (1.f - b); } break;
        case 3: for (size_t i = 0; i < n; ++i) { float b = yp[i] * inv; dxp[i] = dy[i] * s * (1.f - b * b); } break;
        case 4: for (size_t i = 0; i < n; ++i) { float sg = 1.f / (1.f + expf(-xp[i])); dxp[i] = dy[i] * s * sg; } break;
        default: for (size_t i = 0; i < n; ++i) dxp[i] = dy[i] * s; break;
    }
}
