#ifndef BNN_DSP_H
#define BNN_DSP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DSP 后端抽象 (与算子后端并列, 同样可替换硬件实现).
 *  - 默认 CPU: radix-2 迭代 FFT;
 *  - MCU 可用 bnn_dsp_set_backend() 接入 ESP-DSP / CMSIS-DSP 的优化 FFT;
 *  - 复数以交错 (re, im) 存储; rfft 约定与 numpy.rfft 一致 (含 1/n 的 irfft 归一).
 *
 * 注: 谐波/Mel/MFCC/YIN 等更高层算法位于 dsp/ 组, 只调用本后端的 FFT 原语,
 *     从而把"频谱变换"这一硬件相关部分隔离在此处.
 */
typedef struct bnn_dsp_backend {
    const char *name;

    /* 实数 FFT: in[n] 实数 -> out[2*(n/2+1)] 复数(交错). n 必须为 2 的幂. */
    void (*rfft)(const float *in, float *out_complex, int n);

    /* 逆实数 FFT: in[2*(n/2+1)] 复数(交错, 厄米) -> out[n] 实数, 已 /n 归一. */
    void (*irfft)(const float *in_complex, float *out, int n);

    /* 复数 FFT (原地, data[2*n] 交错). dir=-1 正变换(e^{-j..}), dir=+1 逆变换(未归一). */
    void (*cfft)(float *data, int n, int dir);
} bnn_dsp_backend_t;

void                     bnn_dsp_set_backend(const bnn_dsp_backend_t *be);
const bnn_dsp_backend_t *bnn_dsp_get_backend(void);
const bnn_dsp_backend_t *bnn_dsp_cpu_backend(void);

#define BNN_DSP() (bnn_dsp_get_backend())

#ifdef __cplusplus
}
#endif
#endif
