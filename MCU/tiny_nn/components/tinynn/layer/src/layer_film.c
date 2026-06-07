#include "bnn_layer/bnn_layer.h"
#include "bnn_layer/bnn_xform_layers.h"
#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

/*
 * FiLM 条件调制 (双输入, 推理前向):
 *   inputs[0] = 特征图 x : [B, C, T]
 *   inputs[1] = 条件向量 e : [B, E]   (乐器嵌入)
 *   gb[B, 2C] = e[B,E] * W[E,2C] + bias[2C]    (gamma=gb[:, :C], beta=gb[:, C:])
 *   y[b,c,t]  = (1 + gamma[b,c]) * x[b,c,t] + beta[b,c]
 *
 * 权重布局 W:[E,2C] 与 PyTorch nn.Linear(E,2C) 的权重需转置后导出 (见 Python 导出器).
 * backward = NULL (训练在 PyTorch).
 */
typedef struct {
    bnn_layer_t base;
    int C, E;
    float gamma_base;   /* 1.0: (1+gamma)*x+beta;  0.0: gamma*x+beta */
    float *W;   /* [E, 2C] */
    float *b;   /* [2C]   */
    bnn_layer_param_ref_t pref_w, pref_b;
} film_t;

static bnn_layer_t *film_create(const bnn_layer_cfg_t *cfg) {
    if (!cfg) { BNN_LOGE("film: null cfg"); return NULL; }
    int C = 0, E = 0;
    float gamma_base = 1.0f;       /* 默认 DDSP 约定 (1+gamma) */
    if (cfg->extra) {
        const bnn_film_cfg_t *fc = (const bnn_film_cfg_t *)cfg->extra;
        C = fc->channels; E = fc->embedding_dim;
        gamma_base = fc->gamma_plus_one ? 1.0f : 0.0f;
    } else {                       /* 回退: 复用通用字段 */
        C = cfg->out_channels; E = cfg->in_features;
    }
    if (C <= 0 || E <= 0) { BNN_LOGE("film: invalid C/E (C=%d E=%d)", C, E); return NULL; }

    film_t *f = (film_t *)bnn_calloc(1, sizeof(*f));
    if (!f) return NULL;
    f->C = C; f->E = E; f->gamma_base = gamma_base;
    int wn = E * 2 * C;
    f->W = (float *)bnn_calloc(wn, sizeof(float));
    f->b = (float *)bnn_calloc(2 * C, sizeof(float));
    if (!f->W || !f->b) { if (f->W) bnn_free(f->W); if (f->b) bnn_free(f->b); bnn_free(f); return NULL; }
    /* 零初始化 -> 初始恒等 (gamma=0->(1+0), beta=0); 训练后由权重加载覆盖 */
    f->pref_w.data = f->W; f->pref_w.grad = NULL; f->pref_w.numel = (size_t)wn;    f->pref_w.next = &f->pref_b;
    f->pref_b.data = f->b; f->pref_b.grad = NULL; f->pref_b.numel = (size_t)2 * C; f->pref_b.next = NULL;
    return &f->base;
}

static void film_destroy(bnn_layer_t *self) {
    if (!self) return;
    film_t *f = (film_t *)self;
    if (f->W) bnn_free(f->W);
    if (f->b) bnn_free(f->b);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(f);
}

static void film_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    (void)self;
    for (int i = 0; i < in; ++i) os[i] = is[i];   /* 输出形状 = 特征图(dep0) */
    *on = in;
}

static bnn_tensor_t *film_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    if (n_in < 2) { BNN_LOGE("film needs 2 inputs (x, e)"); return NULL; }
    film_t *f = (film_t *)self;
    bnn_tensor_t *x = inputs[0];
    bnn_tensor_t *e = inputs[1];
    int B = x->shape[0], C = x->shape[1], T = x->shape[2];
    if (C != f->C) { BNN_LOGE("film: channel mismatch %d vs %d", C, f->C); return NULL; }

    if (self->cached_output && self->cached_output->numel != x->numel) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, x->ndim, x->shape);
        if (!self->cached_output) return NULL;
    }
    self->cached_input = x;

    const bnn_op_backend_t *op = BNN_OP();
    size_t mark = bnn_ws_mark(NULL);
    float *gb = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)B * 2 * C);
    if (!gb) { BNN_LOGE("film ws OOM"); return NULL; }

    /* gb[B,2C] = e[B,E] * W[E,2C] + b[2C] */
    op->gemm((const float *)e->data, f->W, f->b, gb, B, 2 * C, f->E, 0);

    const float *X = (const float *)x->data;
    float *Y = (float *)self->cached_output->data;
    for (int bi = 0; bi < B; ++bi) {
        const float *gbr = gb + (size_t)bi * 2 * C;
        for (int c = 0; c < C; ++c) {
            float gamma = f->gamma_base + gbr[c];
            float beta  = gbr[C + c];
            const float *xr = X + ((size_t)bi * C + c) * T;
            float *yr       = Y + ((size_t)bi * C + c) * T;
            for (int t = 0; t < T; ++t) yr[t] = gamma * xr[t] + beta;
        }
    }
    bnn_ws_reset_to(NULL, mark);
    return self->cached_output;
}

static bnn_layer_param_ref_t *film_params(bnn_layer_t *self) {
    return &((film_t *)self)->pref_w;
}

static const bnn_layer_vtbl_t film_vtbl = {
    .create = film_create, .destroy = film_destroy, .infer_shape = film_infer,
    .forward = film_forward, .backward = NULL, .params = film_params,
};

BNN_REGISTER_LAYER(film, &film_vtbl)
