#include "bnn_layer/bnn_layer.h"
#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_mem.h"
#include <string.h>

/*
 * Softmax 层 (按 row).  cfg.in_features = num_class (用于 shape 校验, 可不填)
 * 注意: 训练分类一般直接用 softmax_ce loss + 输出 logits, 此层主要用于
 *       推理或自定义训练流程.
 *
 * 反向: dx_i = y_i * (dy_i - sum_j(dy_j * y_j))
 */
typedef struct { bnn_layer_t base; } sm_t;

static bnn_layer_t *sm_create(const bnn_layer_cfg_t *cfg);
static void         sm_destroy(bnn_layer_t *self);
static void         sm_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*sm_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         sm_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                const float *dy, float **dx);
static bnn_layer_param_ref_t *sm_params(bnn_layer_t *self) { (void)self; return NULL; }

static const bnn_layer_vtbl_t sm_vtbl = {
    .create = sm_create, .destroy = sm_destroy, .infer_shape = sm_infer,
    .forward = sm_forward, .backward = sm_backward, .params = sm_params,
};
BNN_REGISTER_LAYER(softmax, &sm_vtbl)

static bnn_layer_t *sm_create(const bnn_layer_cfg_t *cfg) {
    (void)cfg;
    sm_t *s = (sm_t *)bnn_calloc(1, sizeof(*s));
    if (s) s->base.vtbl = &sm_vtbl;
    return &s->base;
}
static void sm_destroy(bnn_layer_t *self) {
    if (!self) return;
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(self);
}
static void sm_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self; for (int i=0;i<in;++i) os[i]=is[i]; *on=in;
}
static bnn_tensor_t *sm_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    bnn_tensor_t *x = inputs[0];
    if (self->cached_output && self->cached_output->numel != x->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, x->ndim, x->shape);
        if (!self->cached_output) return NULL;
    }
    int B = x->shape[0];
    int K = (int)(x->numel / (size_t)B);
    BNN_OP()->softmax_rows((const float *)x->data, (float *)self->cached_output->data, B, K);
    return self->cached_output;
}
static void sm_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                        const float *dy, float **dx) {
    (void)n_in;
    if (!dx || !dx[0]) return;
    bnn_tensor_t *x = inputs[0];
    const float *y = (const float *)self->cached_output->data;
    int B = x->shape[0];
    int K = (int)(x->numel / (size_t)B);
    for (int b = 0; b < B; ++b) {
        const float *yr = y + b*K;
        const float *dyr = dy + b*K;
        float *dxr = dx[0] + b*K;
        float s = 0.f;
        for (int i = 0; i < K; ++i) s += dyr[i] * yr[i];
        for (int i = 0; i < K; ++i) dxr[i] = yr[i] * (dyr[i] - s);
    }
}
