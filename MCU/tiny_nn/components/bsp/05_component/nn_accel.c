/*
 * nn_accel.c — ESP32-P4 NN 算子加速后端
 *
 * Phase 1 (F32): dspm_mult_f32_arp4 / dsps_dotprod / dsps_vec
 * Phase 2 (INT8): esp_nn_conv_s8_esp32p4 (via esp_nn_conv_s8 macro)
 */

#include "nn_accel.h"
#include "bnn_op/bnn_op.h"
#include "bnn_op/bnn_nn.h"
#include "bnn_utils/bnn_workspace.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

#include "dspm_mult.h"
#include "dsps_dotprod.h"
#include "dsps_add.h"
#include "dsps_sub.h"
#include "dsps_mul.h"
#include "dsps_addc.h"
#include "dsps_mulc.h"

#include "esp_nn.h"

#include <math.h>
#include <string.h>
#include <float.h>

static const char *TAG = "nn_accel";
static int  s_ready      = 0;
static int  s_int8_ready = 0;
static void *s_nn_scratch = NULL;
static size_t s_nn_scratch_sz = 0;

/* ── Phase 1: F32 热点算子 (其余字段运行时从 cpu_ref 复制) ───────────────── */

static bnn_op_backend_t g_dsp_backend;

static void op_gemm_dsp(const float *A, const float *B, const float *bias,
                         float *C, int M, int N, int K, int acc)
{
    if (!acc && !bias) {
        dspm_mult_f32(A, B, C, M, K, N);
        return;
    }
    size_t mark = bnn_ws_mark(NULL);
    float *tmp = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)M * (size_t)N);
    if (!tmp) {
        bnn_op_cpu_backend()->gemm(A, B, bias, C, M, N, K, acc);
        return;
    }
    dspm_mult_f32(A, B, tmp, M, K, N);
    if (!acc && bias) {
        for (int i = 0; i < M; ++i)
            dsps_add_f32(tmp + (size_t)i * N, bias, C + (size_t)i * N, N, 1, 1, 1);
    } else if (acc && !bias) {
        for (int i = 0; i < M; ++i)
            dsps_add_f32(C + (size_t)i * N, tmp + (size_t)i * N, C + (size_t)i * N, N, 1, 1, 1);
    } else if (acc && bias) {
        for (int i = 0; i < M; ++i) {
            dsps_add_f32(tmp + (size_t)i * N, bias, tmp + (size_t)i * N, N, 1, 1, 1);
            dsps_add_f32(C + (size_t)i * N, tmp + (size_t)i * N, C + (size_t)i * N, N, 1, 1, 1);
        }
    }
    bnn_ws_reset_to(NULL, mark);
}

/*
 * gemm_nt: C[M,N] = A[M,K] × B[N,K]^T
 *
 * 与 gemm 不同，B 已经是转置布局 [N,K]，因此每次 dot(A[i,:], B[j,:])
 * 两个向量都是连续内存，dsps_dotprod_f32_arp4 可以充分发挥 PIE 硬件循环.
 * 对 conv1d: A=W[Cout,Kall] in PSRAM, B=col_T[To,Kall] in SRAM,
 *   A 行顺序读（PSRAM burst），B 行顺序读（SRAM 极快），消除旧方案 B 列跳读 256B.
 */
static void op_gemm_nt_dsp(const float *A, const float *B, const float *bias,
                            float *C, int M, int N, int K, int acc)
{
    if (!acc) memset(C, 0, sizeof(float) * (size_t)M * (size_t)N);
    for (int i = 0; i < M; ++i) {
        const float *ai = A + (size_t)i * K;
        float *ci = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            float d = 0.f;
            dsps_dotprod_f32(ai, B + (size_t)j * K, &d, K);
            ci[j] += d + (bias ? bias[j] : 0.f);
        }
    }
}

static void op_add_dsp(const float *a, const float *b, float *c, size_t n)
{
    dsps_add_f32(a, b, c, (int)n, 1, 1, 1);
}

static void op_sub_dsp(const float *a, const float *b, float *c, size_t n)
{
    dsps_sub_f32(a, b, c, (int)n, 1, 1, 1);
}

static void op_mul_dsp(const float *a, const float *b, float *c, size_t n)
{
    dsps_mul_f32(a, b, c, (int)n, 1, 1, 1);
}

static void op_adds_dsp(const float *a, float s, float *c, size_t n)
{
    dsps_addc_f32(a, c, (int)n, s, 1, 1);
}

static void op_muls_dsp(const float *a, float s, float *c, size_t n)
{
    dsps_mulc_f32(a, c, (int)n, s, 1, 1);
}

static float op_dot_dsp(const float *a, const float *b, size_t n)
{
    float r = 0.f;
    dsps_dotprod_f32(a, b, &r, (int)n);
    return r;
}

static void nn_accel_build_dsp_backend(void)
{
    g_dsp_backend = *bnn_op_cpu_backend();
    g_dsp_backend.name    = "esp_dsp_p4_arp4";
    g_dsp_backend.gemm    = op_gemm_dsp;
    g_dsp_backend.gemm_nt = op_gemm_nt_dsp;
    g_dsp_backend.add     = op_add_dsp;
    g_dsp_backend.sub     = op_sub_dsp;
    g_dsp_backend.mul     = op_mul_dsp;
    g_dsp_backend.adds    = op_adds_dsp;
    g_dsp_backend.muls    = op_muls_dsp;
    g_dsp_backend.dot     = op_dot_dsp;
}

/* ── Phase 1 公开接口 ─────────────────────────────────────────────────────── */

esp_err_t nn_accel_init(void)
{
    if (s_ready) return ESP_OK;
    nn_accel_build_dsp_backend();
    s_ready = 1;
    ESP_LOGI(TAG, "Phase-1 ready: esp-dsp P4 PIE GEMM/vec (CONFIG_DSP_OPTIMIZED=%d)",
#ifdef CONFIG_DSP_OPTIMIZED
             1
#else
             0
#endif
    );
    return ESP_OK;
}

const bnn_op_backend_t *nn_accel_backend(void)
{
    if (!s_ready) nn_accel_build_dsp_backend();
    return &g_dsp_backend;
}

/* ── Phase 2: ESP-NN INT8 Conv1D ─────────────────────────────────────────── */

static int nn_conv_scratch_size(int T, int Cin, int Cout, int K, int pad)
{
    data_dims_t input_dims  = { .width = T,  .height = 1, .channels = Cin,  .extra = 1 };
    data_dims_t filter_dims = { .width = K,  .height = 1, .channels = Cin,  .extra = 1 };
    data_dims_t output_dims = { .width = T,  .height = 1, .channels = Cout, .extra = 1 };
    conv_params_t conv_params = {
        .in_offset  = 0,
        .out_offset = 0,
        .stride     = { 1, 1 },
        .padding    = { pad, 0 },
        .dilation   = { 1, 1 },
        .activation = { -128, 127 },
    };
    return esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                        &output_dims, &conv_params);
}

static void nn_conv1d_s8_wrapper(
    const int8_t  *input,   int32_t in_offset,
    int T, int Cin, int pad, int stride, int dilation,
    const int8_t  *filter,  int32_t filter_offset,
    int Cout, int K,
    const int32_t *bias,
    int8_t  *output,        int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult,
    int To)
{
    (void)filter_offset;
    nn_accel_conv1d_s8(input, in_offset, T, Cin, pad, stride, dilation,
                       filter, 0, Cout, K, bias, output, out_offset,
                       out_shift, out_mult, To);
}

static const bnn_nn_backend_t g_nn_layer_backend = {
    .name       = "esp_nn_p4_int8",
    .conv1d_s8  = nn_conv1d_s8_wrapper,
};

esp_err_t nn_accel_init_int8(void)
{
    if (s_int8_ready) return ESP_OK;

    /* 按最大 Conv1d 层 (H=128, Cin=128, K=3, T=64) 预分配 scratch */
    int sz = nn_conv_scratch_size(64, 128, 128, 3, 1);
    if (sz < 0) sz = 0;
    if (sz > (int)s_nn_scratch_sz) {
        if (s_nn_scratch) {
            heap_caps_free(s_nn_scratch);
            s_nn_scratch = NULL;
            s_nn_scratch_sz = 0;
        }
        if (sz > 0) {
            s_nn_scratch = heap_caps_malloc((size_t)sz, MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
            if (!s_nn_scratch) {
                ESP_LOGE(TAG, "ESP-NN scratch OOM (需 %d B 内部 RAM)", sz);
                return ESP_ERR_NO_MEM;
            }
            s_nn_scratch_sz = (size_t)sz;
        }
    }
    if (s_nn_scratch && sz > 0) {
        esp_nn_set_conv_scratch_buf(s_nn_scratch);
    }

    bnn_nn_set_backend(&g_nn_layer_backend);
    s_int8_ready = 1;
    ESP_LOGI(TAG, "Phase-2 ready: ESP-NN INT8 conv P4 (scratch=%d B)", sz);
    return ESP_OK;
}

const bnn_nn_backend_t *nn_accel_nn_backend(void)
{
    return &g_nn_layer_backend;
}

int nn_accel_has_int8_conv(void)
{
    return s_int8_ready;
}

void nn_accel_conv1d_s8(
    const int8_t  *input,   int32_t in_offset,
    int T, int Cin, int pad, int stride, int dilation,
    const int8_t  *filter,  int32_t filter_offset,
    int Cout, int K,
    const int32_t *bias,
    int8_t  *output,        int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult,
    int To)
{
    (void)filter_offset;
    if (!s_int8_ready || dilation != 1 || stride != 1) return;

    data_dims_t input_dims  = { .width = T,  .height = 1, .channels = Cin,  .extra = 1 };
    data_dims_t filter_dims = { .width = K,  .height = 1, .channels = Cin,  .extra = 1 };
    data_dims_t output_dims = { .width = To, .height = 1, .channels = Cout, .extra = 1 };
    conv_params_t conv_params = {
        .in_offset  = in_offset,
        .out_offset = out_offset,
        .stride     = { (int32_t)stride, 1 },
        .padding    = { pad, 0 },
        .dilation   = { 1, 1 },
        .activation = { -128, 127 },
    };
    quant_data_t quant_data = {
        .shift = (int32_t *)out_shift,
        .mult  = (int32_t *)out_mult,
    };

    /* 输入 layout: tinynn [Cin,T] → ESP-NN NHWC [T,Cin] (原地暂用 workspace) */
    size_t mark = bnn_ws_mark(NULL);
    int8_t *in_nhwc = (int8_t *)bnn_ws_alloc(NULL, (size_t)T * Cin);
    int8_t *out_nhwc = (int8_t *)bnn_ws_alloc(NULL, (size_t)To * Cout);
    if (!in_nhwc || !out_nhwc) {
        bnn_ws_reset_to(NULL, mark);
        return;
    }
    for (int t = 0; t < T; ++t)
        for (int ci = 0; ci < Cin; ++ci)
            in_nhwc[(size_t)t * Cin + ci] = input[(size_t)ci * T + t];

    int need_scratch = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                                    &output_dims, &conv_params);
    if (need_scratch > (int)s_nn_scratch_sz) {
        bnn_ws_reset_to(NULL, mark);
        return;
    }
    if (need_scratch > 0 && s_nn_scratch) {
        esp_nn_set_conv_scratch_buf(s_nn_scratch);
    }

    esp_nn_conv_s8(&input_dims, in_nhwc, &filter_dims, filter,
                   bias, &output_dims, out_nhwc, &conv_params, &quant_data);

    for (int t = 0; t < To; ++t)
        for (int oc = 0; oc < Cout; ++oc)
            output[(size_t)oc * To + t] = out_nhwc[(size_t)t * Cout + oc];

    bnn_ws_reset_to(NULL, mark);
}
