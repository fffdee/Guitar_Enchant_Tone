/* ESP32-P4 NN 算子加速后端: 用 esp-dsp PIE 向量指令替换 tinynn cpu_ref 实现.
 *
 * Phase 1 (F32 加速):
 *   - gemm       → dspm_mult_f32_arp4  (P4 128-bit PIE GEMM, conv1d 主热点)
 *   - dot        → dsps_dotprod_f32_arp4
 *   - add/sub/mul → dsps_xxx_f32_arp4  (按位步进=1 的连续向量)
 *   - adds/muls  → dsps_addc/mulc_f32_arp4
 *   - 其余算子   → fallback bnn_op_cpu_backend()
 *
 * Phase 2 (INT8 ESP-NN):
 *   - nn_accel_conv1d_s8(): 用 esp_nn_conv_opt_s8 (H=1) 加速量化卷积
 *   - nn_accel_init_int8(): 分配 ESP-NN scratch buffer
 *
 * 用法:
 *   nn_accel_init() / nn_accel_init_int8()
 *   bnn_op_set_backend(nn_accel_backend())
 */
#ifndef NN_ACCEL_H
#define NN_ACCEL_H

#include "esp_err.h"
#include "bnn_op/bnn_op.h"
#include "bnn_op/bnn_nn.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Phase 1: F32 PIE GEMM/向量后端 ──────────────────────────────────────── */

/* 初始化 (当前无状态; 预留供未来预分配 scratch). */
esp_err_t nn_accel_init(void);

/* 取 ESP-DSP F32 加速的 bnn_op_backend_t. 须先调用 nn_accel_init(). */
const bnn_op_backend_t *nn_accel_backend(void);

/* 取 ESP-NN INT8 加速的 bnn_nn_backend_t. 须先调用 nn_accel_init_int8(). */
const bnn_nn_backend_t *nn_accel_nn_backend(void);

/* ── Phase 2: INT8 ESP-NN 卷积接口 ──────────────────────────────────────── */

/* 初始化 ESP-NN scratch buffer (内部 RAM). 须在 nn_accel_init() 之后调用. */
esp_err_t nn_accel_init_int8(void);

/* 查询 INT8 卷积是否可用 (已初始化且 scratch 充足). */
int nn_accel_has_int8_conv(void);

/*
 * Conv1D INT8 推理 (通过 H=1 的 esp_nn_conv_opt_s8 实现).
 *
 *  input [Cin, T]     int8, NHWC 内部重排由调用方完成
 *  filter [Cout, Cin, K] int8
 *  bias   [Cout]         int32
 *  output [Cout, To]  int8
 *  out_shift [Cout]  per-channel 右移量 (定点反量化)
 *  out_mult  [Cout]  per-channel 乘子 (定点反量化), 可为 NULL
 */
void nn_accel_conv1d_s8(
    const int8_t  *input,   int32_t in_offset,
    int T, int Cin, int pad, int stride, int dilation,
    const int8_t  *filter,  int32_t filter_offset,
    int Cout, int K,
    const int32_t *bias,
    int8_t  *output,        int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult,
    int To
);

#ifdef __cplusplus
}
#endif
#endif /* NN_ACCEL_H */
