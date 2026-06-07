#include "bnn_layer/bnn_layer.h"
#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

/*
 * Conv1D (膨胀, same/任意 padding) —— 推理前向 (im2col-1d + GEMM).
 *  输入  X : [B, Cin, T]
 *  权重  W : [Cout, Cin, K]  (展平为 [Cout, Cin*K], 与 PyTorch conv1d 权重内存布局一致)
 *  偏置  b : [Cout]
 *  输出  Y : [B, Cout, To],  To = (T + 2P - D*(K-1) - 1)/S + 1
 *
 *  col[Cin*K, To]:  row=ci*K+k, 列 t -> 取 X[ci, t*S - P + k*D] (越界补零)
 *  Y[Cout, To] = W[Cout, Cin*K] * col + bias_row
 *
 *  训练在 PyTorch 完成: 此处 backward = NULL (graph 自动跳过), 仅保留 params 供权重加载.
 */
typedef struct {
    bnn_layer_t base;
    int Cin, Cout, K, S, P, D;
    int Tin_cached, To_cached;
    float *W, *b;
    bnn_layer_param_ref_t pref_w, pref_b;
} conv1d_t;

static void rand_init(float *p, int fan_in, int n) {
    static unsigned int s = 0x6d2b79f5u;
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = ((s >> 8) & 0xFFFFFF) / (float)0x1000000;
        p[i] = (u * 2.f - 1.f) * scale;
    }
}

static int conv1d_out_len(int T, int K, int S, int P, int D) {
    return (T + 2 * P - D * (K - 1) - 1) / S + 1;
}

static bnn_layer_t *conv1d_create(const bnn_layer_cfg_t *cfg) {
    if (!cfg || cfg->in_channels <= 0 || cfg->out_channels <= 0 || cfg->kernel <= 0) {
        BNN_LOGE("conv1d: invalid cfg"); return NULL;
    }
    conv1d_t *c = (conv1d_t *)bnn_calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->Cin = cfg->in_channels; c->Cout = cfg->out_channels;
    c->K = cfg->kernel;
    c->S = cfg->stride   > 0 ? cfg->stride   : 1;
    c->D = cfg->dilation > 0 ? cfg->dilation : 1;
    c->P = cfg->padding;
    int wn = c->Cout * c->Cin * c->K;
    c->W = (float *)bnn_calloc(wn, sizeof(float));
    c->b = (float *)bnn_calloc(c->Cout, sizeof(float));
    if (!c->W || !c->b) { if (c->W) bnn_free(c->W); if (c->b) bnn_free(c->b); bnn_free(c); return NULL; }
    rand_init(c->W, c->Cin * c->K, wn);
    c->pref_w.data = c->W; c->pref_w.grad = NULL; c->pref_w.numel = (size_t)wn;   c->pref_w.next = &c->pref_b;
    c->pref_b.data = c->b; c->pref_b.grad = NULL; c->pref_b.numel = (size_t)c->Cout; c->pref_b.next = NULL;
    return &c->base;
}

static void conv1d_destroy(bnn_layer_t *self) {
    if (!self) return;
    conv1d_t *c = (conv1d_t *)self;
    if (c->W) bnn_free(c->W);
    if (c->b) bnn_free(c->b);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(c);
}

static void conv1d_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    conv1d_t *c = (conv1d_t *)self; (void)in;
    int B = is[0], T = is[2];
    os[0] = B; os[1] = c->Cout; os[2] = conv1d_out_len(T, c->K, c->S, c->P, c->D);
    *on = 3;
}

static bnn_tensor_t *conv1d_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    conv1d_t *c = (conv1d_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0], T = x->shape[2];
    int To = conv1d_out_len(T, c->K, c->S, c->P, c->D);
    c->Tin_cached = T; c->To_cached = To;
    int os[3] = { B, c->Cout, To };

    if (self->cached_output && (self->cached_output->shape[0] != B ||
                                self->cached_output->shape[2] != To)) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 3, os);
        if (!self->cached_output) return NULL;
    }
    self->cached_input = x;

    const bnn_op_backend_t *op = BNN_OP();
    int Kall = c->Cin * c->K;
    size_t mark = bnn_ws_mark(NULL);
    float *col = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)Kall * To);
    if (!col) { BNN_LOGE("conv1d ws OOM"); return NULL; }

    const float *X = (const float *)x->data;
    float *Y = (float *)self->cached_output->data;
    for (int bi = 0; bi < B; ++bi) {
        const float *Xb = X + (size_t)bi * c->Cin * T;
        /* im2col-1d */
        for (int ci = 0; ci < c->Cin; ++ci) {
            const float *xrow = Xb + (size_t)ci * T;
            for (int k = 0; k < c->K; ++k) {
                float *crow = col + (size_t)(ci * c->K + k) * To;
                int base = k * c->D - c->P;
                for (int t = 0; t < To; ++t) {
                    int idx = t * c->S + base;
                    crow[t] = (idx >= 0 && idx < T) ? xrow[idx] : 0.f;
                }
            }
        }
        float *Yb = Y + (size_t)bi * c->Cout * To;
        op->gemm(c->W, col, NULL, Yb, c->Cout, To, Kall, 0);
        for (int oc = 0; oc < c->Cout; ++oc) {
            float bv = c->b[oc];
            float *row = Yb + (size_t)oc * To;
            for (int t = 0; t < To; ++t) row[t] += bv;
        }
    }
    bnn_ws_reset_to(NULL, mark);
    return self->cached_output;
}

static bnn_layer_param_ref_t *conv1d_params(bnn_layer_t *self) {
    return &((conv1d_t *)self)->pref_w;
}

static const bnn_layer_vtbl_t conv1d_vtbl = {
    .create = conv1d_create, .destroy = conv1d_destroy, .infer_shape = conv1d_infer,
    .forward = conv1d_forward, .backward = NULL, .params = conv1d_params,
};

BNN_REGISTER_LAYER(conv1d, &conv1d_vtbl)
