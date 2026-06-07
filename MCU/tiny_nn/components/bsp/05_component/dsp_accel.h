/* ESP32-P4 加速 DSP 后端: 用 esp-dsp 的单精度 FFT (arp4 汇编) 替换 tinynn 默认
 * 的 double-twiddle radix-2 FFT. P4 FPU 仅单精度, 默认实现的 double 递推走软件
 * 模拟极慢; 接入此后端后 FFT 走硬件 FPU + 预计算表, 大幅提速.
 *
 * 用法: 先 dsp_accel_init(max_n_fft), 再 bnn_dsp_set_backend(dsp_accel_backend()).
 */
#ifndef DSP_ACCEL_H
#define DSP_ACCEL_H

#include "esp_err.h"
#include "bnn_op/bnn_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化 esp-dsp twiddle 表 + 内部 scratch (max_n = 最大 FFT 点数, 2 的幂). */
esp_err_t dsp_accel_init(int max_n);

/* 取 esp-dsp 实现的 bnn DSP 后端 (rfft/irfft/cfft). 须先 dsp_accel_init. */
const bnn_dsp_backend_t *dsp_accel_backend(void);

#ifdef __cplusplus
}
#endif
#endif
