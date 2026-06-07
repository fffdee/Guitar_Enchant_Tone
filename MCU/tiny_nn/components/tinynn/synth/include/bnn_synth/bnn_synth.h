#ifndef BNN_SYNTH_H
#define BNN_SYNTH_H

#include "bnn_frontend/bnn_xform_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DDSP 合成器 (前端的"输出侧"对偶, 与 ddsp/synthesizer.py 的 numpy 版对齐):
 *   谐波加性振荡器 (相位累加) + 子带滤波噪声 (单位 RMS 带通噪声 × 逐帧增益).
 *
 * 面向对象: 不透明对象 + create/destroy/render, 频谱变换委托 op 层的 bnn_dsp FFT 后端
 * (因此 FFT 可被 ESP-DSP / CMSIS-DSP 替换, 合成器本身与硬件解耦).
 *
 * 参数布局 (与训练标签一致):
 *   harmonic_amp : 逐帧 [T][n_harmonics]  谐波幅度 (时域正弦幅度)
 *   noise_band   : 逐帧 [T][n_noise_bands] 各带时域 RMS
 *   params       : 逐帧 [T][n_harmonics + n_noise_bands], 前 K 谐波, 后 B 噪声
 *   f0_hz        : 逐帧 [T] 基频 (来自前端)
 * 输出音频长度 n_samples (默认 T*hop_size).
 *
 * 说明: 噪声路径用整段 (补零到 2 的幂) FFT 做理想带通, 与 Python 算法一致;
 *       具体白噪声实现与 numpy RNG 不同, 故噪声为统计等价(各带 RMS 受控)而非逐样点一致,
 *       谐波路径与 Python 数值高度一致. 大段合成内存随 n_samples 增长, MCU 宜分块调用.
 */
typedef struct bnn_synth bnn_synth_t;

bnn_synth_t *bnn_synth_create(const bnn_xform_cfg_t *cfg);
void         bnn_synth_destroy(bnn_synth_t *s);

/* 设定噪声 RNG 种子 (默认 0, 与 Python synth seed=0 对应, 保证可复现) */
void bnn_synth_set_seed(bnn_synth_t *s, unsigned int seed);

/* 查询噪声子带 bin 边界 (长度 n_noise_bands+1). 返回指针, *n_edges 写边界数; 失败返回 NULL. */
const int *bnn_synth_band_edges(const bnn_synth_t *s, int *n_edges);

/* 谐波加性合成: 写入 out[0..n_samples). 成功返回 0. n_samples<=0 时取 T*hop. */
int bnn_synth_harmonic(bnn_synth_t *s, const float *f0_hz,
                       const float *harmonic_amp, int T,
                       float *out, int n_samples);

/* 子带滤波噪声: 累加到 out[0..n_samples) (便于先谐波后噪声叠加). 成功返回 0. */
int bnn_synth_noise(bnn_synth_t *s, const float *noise_band, int T,
                    float *out, int n_samples);

/* 完整 DDSP: 内部先谐波(写)再噪声(加). params 为逐帧 [T][K+B]. 成功返回 0. */
int bnn_synth_render(bnn_synth_t *s, const float *f0_hz, const float *params,
                     int T, float *out, int n_samples);

#ifdef __cplusplus
}
#endif
#endif
