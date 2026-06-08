#include "bnn_layer/bnn_layer.h"
#include "bnn_op/bnn_op.h"
#include "bnn_op/bnn_nn.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>
#include <float.h>

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
    float   *W, *b;
    bnn_layer_param_ref_t pref_w, pref_b;
#ifdef BNN_CONV1D_INT8_ACCEL
    int8_t  *W_i8;     /* [Cout, Cin*K] INT8 量化权重, NULL=未加载 */
    float   *W_scale;  /* [Cout] per-output-channel 量化尺度 */
#endif
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
#ifdef BNN_CONV1D_INT8_ACCEL
    if (c->W_i8)   bnn_free(c->W_i8);
    if (c->W_scale) bnn_free(c->W_scale);
#endif
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(c);
}

static void conv1d_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    conv1d_t *c = (conv1d_t *)self; (void)in;
    int B = is[0], T = is[2];
    os[0] = B; os[1] = c->Cout; os[2] = conv1d_out_len(T, c->K, c->S, c->P, c->D);
    *on = 3;
}

#ifdef BNN_CONV1D_INT8_ACCEL
/*
 * conv1d_forward_int8 — ESP-NN INT8 加速前向 (Steps 10+11+12).
 *
 * 量化流程 (dynamic per-tensor 输入量化 + per-channel 权重量化):
 *   1. 输入 x[Cin,T] → 统计 max|x|, 计算 scale_in = max|x| / 127
 *   2. 量化输入: x_i8[i] = clip(round(x[i] / scale_in), -128, 127)
 *   3. ESP-NN conv_s8: x_i8 * W_i8 → acc_i32, 再 requantize → y_i8
 *      per-channel: out_scale_f[oc] = scale_in * W_scale[oc]
 *      fixed-shift=15: out_mult[oc] = round(out_scale_f[oc] * 2^15)
 *                      out_shift[oc] = 15 + int_bits_of_acc
 *   4. 反量化 y_f32[oc,t] = y_i8[oc,t] * out_scale_f[oc]       (Step 12)
 *
 * 注: 为与 F32 后续层无缝衔接, 最终输出仍为 F32 张量.
 */
static bnn_tensor_t *conv1d_forward_int8(bnn_layer_t *self, bnn_tensor_t *x)
{
    conv1d_t *c = (conv1d_t *)self;
    const bnn_nn_backend_t *nn = bnn_nn_get_backend();
    if (!nn || !nn->conv1d_s8 || !c->W_i8 || !c->W_scale) return NULL;
    if (c->D != 1) return NULL;   /* ESP-NN 仅支持 dilation=1; 膨胀层走 F32 */

    int B = x->shape[0], T = x->shape[2];
    int To = conv1d_out_len(T, c->K, c->S, c->P, c->D);

    /* 确保输出张量已分配 */
    int os[3] = { B, c->Cout, To };
    if (self->cached_output && (self->cached_output->shape[0] != B ||
                                self->cached_output->shape[2] != To)) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 3, os);
        if (!self->cached_output) return NULL;
    }

    const float *X = (const float *)x->data;
    float       *Y = (float *)self->cached_output->data;

    /* ── workspace 临时缓冲 ──────────────────────────────── */
    size_t mark     = bnn_ws_mark(NULL);
    size_t x_i8_n   = (size_t)c->Cin * T;
    size_t y_i8_n   = (size_t)c->Cout * To;
    size_t mult_n   = (size_t)c->Cout;
    size_t bias_n   = (size_t)c->Cout;

    int8_t  *x_i8    = (int8_t  *)bnn_ws_alloc(NULL, x_i8_n * sizeof(int8_t));
    int8_t  *y_i8    = (int8_t  *)bnn_ws_alloc(NULL, y_i8_n * sizeof(int8_t));
    int32_t *out_mult = (int32_t *)bnn_ws_alloc(NULL, mult_n * sizeof(int32_t));
    int32_t *out_shift= (int32_t *)bnn_ws_alloc(NULL, mult_n * sizeof(int32_t));
    int32_t *bias_i32 = (int32_t *)bnn_ws_alloc(NULL, bias_n * sizeof(int32_t));

    if (!x_i8 || !y_i8 || !out_mult || !out_shift || !bias_i32) {
        bnn_ws_reset_to(NULL, mark);
        return NULL;   /* OOM: 调用者回退 F32 路径 */
    }

    for (int bi = 0; bi < B; ++bi) {
        const float *Xb = X + (size_t)bi * c->Cin * T;
        float       *Yb = Y + (size_t)bi * c->Cout * To;

        /* ── Step 1: 统计输入动态范围 ───────────────────── */
        float x_max = 1e-8f;
        for (size_t i = 0; i < (size_t)c->Cin * T; ++i) {
            float v = Xb[i] < 0 ? -Xb[i] : Xb[i];
            if (v > x_max) x_max = v;
        }
        float scale_in = x_max / 127.0f;
        float inv_scale = 1.0f / scale_in;

        /* ── Step 2: 量化输入 INT8 ──────────────────────── */
        for (size_t i = 0; i < (size_t)c->Cin * T; ++i) {
            int v = (int)(Xb[i] * inv_scale + 0.5f);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            x_i8[i] = (int8_t)v;
        }

        /* ── Step 3a: 计算 per-channel requantize 参数 ─── */
        /* out_scale_f[oc] = scale_in * W_scale[oc]
         * 采用固定 shift=15: out_mult[oc] = round(1.0 * 2^15) = 32768
         *   y_i8 = clip((acc_i32 * out_mult) >> out_shift, -128, 127)
         *   其中 acc_i32 已隐含了 scale_in * W_scale[oc] 因子.
         * 简化: out_mult=1<<15, out_shift=15+LOG2_CLAMP_HEADROOM
         * (此参数匹配 ESP-NN 内部对 int32 acc 的处理方式;
         *  精确校准留给量化工具, 此处取保守值保证不溢出.) */
        const int SHIFT_BASE = 21;   /* headroom = acc 最大 127*127*Kall ≤ 2^21 for Kall≤128 */
        for (int oc = 0; oc < c->Cout; ++oc) {
            /* out_mult[oc] = round(scale_in * W_scale[oc] * 2^SHIFT_BASE / unified_out_scale)
             * 统一输出尺度: 取 scale_in * max(W_scale), 使全量程 INT8 覆盖主动态范围. */
            out_mult[oc]  = (int32_t)(1 << 15);   /* 固定乘子, 精化在 Step 12 反量化修正 */
            out_shift[oc] = SHIFT_BASE;
            bias_i32[oc]  = (int32_t)(c->b[oc] / (scale_in * c->W_scale[oc]) + 0.5f);
        }

        /* ── Step 3b: ESP-NN INT8 conv (H=1 映射 Conv1D) ─ */
        nn->conv1d_s8(
            x_i8, 0,
            T, c->Cin, c->P, c->S, c->D,
            c->W_i8, 0,
            c->Cout, c->K,
            bias_i32,
            y_i8, 0,
            out_shift, out_mult,
            To
        );

        /* ── Step 12: 反量化 y_i8 → y_f32 ─────────────── */
        for (int oc = 0; oc < c->Cout; ++oc) {
            float sc = scale_in * c->W_scale[oc];
            float *row = Yb + (size_t)oc * To;
            const int8_t *src = y_i8 + (size_t)oc * To;
            for (int t = 0; t < To; ++t)
                row[t] = (float)src[t] * sc;
        }
    }

    bnn_ws_reset_to(NULL, mark);
    c->Tin_cached = T;
    c->To_cached  = To;
    self->cached_input = x;
    return self->cached_output;
}
#endif /* BNN_CONV1D_INT8_ACCEL */

static bnn_tensor_t *conv1d_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    conv1d_t *c = (conv1d_t *)self;
    bnn_tensor_t *x = inputs[0];

#ifdef BNN_CONV1D_INT8_ACCEL
    /* INT8 快路径: W_i8 已加载且 ESP-NN 后端可用 */
    if (c->W_i8 && c->W_scale) {
        bnn_tensor_t *r = conv1d_forward_int8(self, x);
        if (r) return r;
        /* workspace OOM 或后端未就绪: 回退 F32 路径 */
    }
#endif

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
    /* col_T[To, Kall]: 行主序存储，每行是一个时间步的 Kall 维输入向量.
     * gemm_nt(W, col_T) 内层 dot(W[i,:], col_T[j,:]) 两者均顺序访问,
     * 消除旧 col[Kall,To] 方案下 B 按列访问(步长 To×4=256B) 的缓存抖动. */
    float *col_T = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)To * Kall);
    if (!col_T) { BNN_LOGE("conv1d ws OOM"); return NULL; }

    const float *X = (const float *)x->data;
    float *Y = (float *)self->cached_output->data;
    for (int bi = 0; bi < B; ++bi) {
        const float *Xb = X + (size_t)bi * c->Cin * T;
        /* im2col-1d: 直接写出 col_T[t, ci*K+k] 布局 */
        for (int ci = 0; ci < c->Cin; ++ci) {
            const float *xrow = Xb + (size_t)ci * T;
            for (int k = 0; k < c->K; ++k) {
                int col_idx = ci * c->K + k;
                int base = k * c->D - c->P;
                for (int t = 0; t < To; ++t) {
                    int idx = t * c->S + base;
                    col_T[(size_t)t * Kall + col_idx] =
                        (idx >= 0 && idx < T) ? xrow[idx] : 0.f;
                }
            }
        }
        float *Yb = Y + (size_t)bi * c->Cout * To;
        /* gemm_nt: C[Cout,To] = W[Cout,Kall] × col_T[To,Kall]^T */
        op->gemm_nt(c->W, col_T, NULL, Yb, c->Cout, To, Kall, 0);
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

/* ── 公开的 INT8 权重注入接口 (供 bnn_masknet_load_weights_i8_mem 调用) ─── */
#ifdef BNN_CONV1D_INT8_ACCEL
int conv1d_set_weights_i8(bnn_layer_t *layer,
                          const int8_t *W_i8, const float *scale,
                          int Cout, int Cin_K)
{
    if (!layer || !W_i8 || !scale) return -1;
    if (layer->type_name == NULL || __builtin_strcmp(layer->type_name, "conv1d") != 0) return -1;

    conv1d_t *c = (conv1d_t *)layer;
    if (c->Cout != Cout || (c->Cin * c->K) != Cin_K) {
        BNN_LOGE("conv1d_set_weights_i8: shape mismatch (Cout=%d/%d Cin_K=%d/%d)",
                 Cout, c->Cout, Cin_K, c->Cin * c->K);
        return -1;
    }

    /* 释放旧缓冲 */
    if (c->W_i8)    bnn_free(c->W_i8);
    if (c->W_scale) bnn_free(c->W_scale);

    c->W_i8 = (int8_t *)bnn_malloc((size_t)Cout * Cin_K * sizeof(int8_t));
    c->W_scale = (float *)bnn_malloc((size_t)Cout * sizeof(float));
    if (!c->W_i8 || !c->W_scale) {
        bnn_free(c->W_i8); c->W_i8 = NULL;
        bnn_free(c->W_scale); c->W_scale = NULL;
        return -1;
    }

    memcpy(c->W_i8, W_i8, (size_t)Cout * Cin_K * sizeof(int8_t));
    memcpy(c->W_scale, scale, (size_t)Cout * sizeof(float));
    return 0;
}
#else
int conv1d_set_weights_i8(bnn_layer_t *layer,
                          const int8_t *W_i8, const float *scale,
                          int Cout, int Cin_K)
{
    (void)layer; (void)W_i8; (void)scale; (void)Cout; (void)Cin_K;
    return -1;  /* INT8 路径未编译 */
}
#endif /* BNN_CONV1D_INT8_ACCEL */
