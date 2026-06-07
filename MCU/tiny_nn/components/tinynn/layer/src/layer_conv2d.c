#include "bnn_layer/bnn_layer.h"
#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

/*
 * Conv2D 高性能实现 (im2col + GEMM).
 *  W reshape -> [Cout, Cin*K*K]
 *  X im2col  -> [Cin*K*K, Ho*Wo]
 *  Y[Cout, Ho*Wo] = W * Xcol  (+ bias broadcast on rows)
 *
 *  Backward:
 *   dW   += dY * Xcol^T          (gemm_nt)
 *   dXcol = W^T * dY             (gemm_tn)
 *   db   += sum_{spatial}(dY)
 *   dX   = col2im(dXcol)
 */

typedef struct {
    bnn_layer_t base;
    int Cin, Cout, K, S, P;
    int Hin_cached, Win_cached;
    int Ho_cached, Wo_cached;
    float *W, *b, *dW, *db;
    bnn_layer_param_ref_t pref_w, pref_b;
} conv_t;

static void rand_init(float *p, int fan_in, int n) {
    static unsigned int s = 0x9e3779b9u;
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = ((s >> 8) & 0xFFFFFF) / (float)0x1000000;
        p[i] = (u * 2.f - 1.f) * scale;
    }
}

static bnn_layer_t *conv_create(const bnn_layer_cfg_t *cfg);
static void         conv_destroy(bnn_layer_t *self);
static void         conv_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on);
static bnn_tensor_t*conv_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
static void         conv_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                                  const float *dy, float **dx);
static bnn_layer_param_ref_t *conv_params(bnn_layer_t *self);

static const bnn_layer_vtbl_t conv_vtbl = {
    .create = conv_create, .destroy = conv_destroy, .infer_shape = conv_infer,
    .forward = conv_forward, .backward = conv_backward, .params = conv_params,
};

BNN_REGISTER_LAYER(conv2d, &conv_vtbl)

static bnn_layer_t *conv_create(const bnn_layer_cfg_t *cfg) {
    if (!cfg || cfg->in_channels <= 0 || cfg->out_channels <= 0 || cfg->kernel <= 0) {
        BNN_LOGE("conv2d: invalid cfg"); return NULL;
    }
    conv_t *c = (conv_t *)bnn_calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->base.vtbl = &conv_vtbl;
    c->Cin = cfg->in_channels; c->Cout = cfg->out_channels;
    c->K   = cfg->kernel;      c->S    = cfg->stride > 0 ? cfg->stride : 1;
    c->P   = cfg->padding;
    int wn = c->Cout * c->Cin * c->K * c->K;
    c->W  = (float *)bnn_calloc(wn, sizeof(float));
    c->b  = (float *)bnn_calloc(c->Cout, sizeof(float));
    c->dW = (float *)bnn_calloc(wn, sizeof(float));
    c->db = (float *)bnn_calloc(c->Cout, sizeof(float));
    if (!c->W || !c->b || !c->dW || !c->db) { conv_destroy(&c->base); return NULL; }
    rand_init(c->W, c->Cin * c->K * c->K, wn);
    c->pref_w.data = c->W; c->pref_w.grad = c->dW; c->pref_w.numel = (size_t)wn;
    c->pref_w.next = &c->pref_b;
    c->pref_b.data = c->b; c->pref_b.grad = c->db; c->pref_b.numel = (size_t)c->Cout;
    c->pref_b.next = NULL;
    return &c->base;
}

static void conv_destroy(bnn_layer_t *self) {
    if (!self) return;
    conv_t *c = (conv_t *)self;
    if (c->W) bnn_free(c->W);
    if (c->b) bnn_free(c->b);
    if (c->dW) bnn_free(c->dW);
    if (c->db) bnn_free(c->db);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(c);
}

static void conv_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    conv_t *c = (conv_t *)self; (void)in;
    int B = is[0], H = is[2], W = is[3];
    int Ho = (H + 2 * c->P - c->K) / c->S + 1;
    int Wo = (W + 2 * c->P - c->K) / c->S + 1;
    os[0] = B; os[1] = c->Cout; os[2] = Ho; os[3] = Wo; *on = 4;
}

static bnn_tensor_t *conv_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    conv_t *c = (conv_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0], H = x->shape[2], W = x->shape[3];
    int Ho = (H + 2 * c->P - c->K) / c->S + 1;
    int Wo = (W + 2 * c->P - c->K) / c->S + 1;
    int os[4] = { B, c->Cout, Ho, Wo };
    c->Hin_cached = H; c->Win_cached = W; c->Ho_cached = Ho; c->Wo_cached = Wo;
    if (self->cached_output && (self->cached_output->shape[0] != B ||
                                self->cached_output->shape[2] != Ho ||
                                self->cached_output->shape[3] != Wo)) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 4, os);
        if (!self->cached_output) return NULL;
    }
    self->cached_input = x;

    const bnn_op_backend_t *op = BNN_OP();
    int Kall = c->Cin * c->K * c->K;
    int cols = Ho * Wo;
    /* 单 batch 复用 im2col 缓冲, 减小峰值内存 */
    size_t mark = bnn_ws_mark(NULL);
    float *col = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)Kall * cols);
    if (!col) { BNN_LOGE("conv2d ws OOM"); return NULL; }

    const float *X = (const float *)x->data;
    float *Y = (float *)self->cached_output->data;
    for (int b = 0; b < B; ++b) {
        op->im2col(X + (size_t)b * c->Cin * H * W,
                   c->Cin, H, W, c->K, c->S, c->P, col);
        /* Y[Cout, cols] = W[Cout, Kall] * col[Kall, cols] + bias_per_row */
        float *Yb = Y + (size_t)b * c->Cout * cols;
        op->gemm(c->W, col, NULL, Yb, c->Cout, cols, Kall, 0);
        /* bias broadcast on rows */
        for (int oc = 0; oc < c->Cout; ++oc) {
            float bv = c->b[oc];
            float *row = Yb + oc * cols;
            for (int i = 0; i < cols; ++i) row[i] += bv;
        }
    }
    bnn_ws_reset_to(NULL, mark);
    return self->cached_output;
}

static void conv_backward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                          const float *dy, float **dx) {
    (void)n_in;
    conv_t *c = (conv_t *)self;
    bnn_tensor_t *x = inputs[0];
    int B = x->shape[0], H = c->Hin_cached, W = c->Win_cached;
    int Ho = c->Ho_cached, Wo = c->Wo_cached;
    int Kall = c->Cin * c->K * c->K;
    int cols = Ho * Wo;
    const bnn_op_backend_t *op = BNN_OP();
    const float *X = (const float *)x->data;
    if (dx && dx[0]) memset(dx[0], 0, sizeof(float) * B * c->Cin * H * W);

    size_t mark = bnn_ws_mark(NULL);
    /* 单次 alloc 拿 (col + dcol), 避免多次 alloc 期间 ws 扩容使旧指针失效 */
    size_t one = sizeof(float) * (size_t)Kall * cols;
    char *block = (char *)bnn_ws_alloc(NULL, one * 2);
    if (!block) { BNN_LOGE("conv bw ws OOM"); return; }
    float *col  = (float *)block;
    float *dcol = (float *)(block + one);

    for (int b = 0; b < B; ++b) {
        const float *dyb = dy + (size_t)b * c->Cout * cols;
        /* db += sum spatial */
        for (int oc = 0; oc < c->Cout; ++oc) {
            const float *row = dyb + oc * cols;
            float s = 0.f;
            for (int i = 0; i < cols; ++i) s += row[i];
            c->db[oc] += s;
        }
        /* dW += dY[Cout, cols] * col[Kall, cols]^T  -> gemm_nt with B as col^T */
        op->im2col(X + (size_t)b * c->Cin * H * W,
                   c->Cin, H, W, c->K, c->S, c->P, col);
        /* dW[Cout, Kall] += dY * col^T : M=Cout N=Kall K=cols  using gemm_nt with B=col[Kall,cols] */
        op->gemm_nt(dyb, col, NULL, c->dW, c->Cout, Kall, cols, 1);

        if (dx && dx[0]) {
            /* dcol[Kall, cols] = W^T[Kall, Cout] * dY[Cout, cols] */
            /* 用 gemm_tn: A=W[Cout, Kall] (treat as [K=Cout, M=Kall]^T), B=dY[Cout, cols] */
            op->gemm_tn(c->W, dyb, NULL, dcol, Kall, cols, c->Cout, 0);
            op->col2im(dcol, c->Cin, H, W, c->K, c->S, c->P,
                       dx[0] + (size_t)b * c->Cin * H * W);
        }
    }
    bnn_ws_reset_to(NULL, mark);
}

static bnn_layer_param_ref_t *conv_params(bnn_layer_t *self) {
    return &((conv_t *)self)->pref_w;
}
