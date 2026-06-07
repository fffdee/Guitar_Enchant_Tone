#ifndef BNN_SPECMAT_H
#define BNN_SPECMAT_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 频谱展开矩阵构建 (与 PC 端 src/esp_xform/mask/audio.py 逐一镜像, 保证数值一致).
 * 放在算子层: 纯数学, 供 frontend(specfront) 与 synth(specsynth) 共用, 框架自洽无需外部矩阵文件.
 *
 * 约定: rfft 频点数 n_bins = n_fft/2+1; HTK 梅尔; fft_freqs = linspace(0, sr/2, n_bins).
 */

/* 梅尔三角滤波器组 Mel[n_mels, n_bins] (行优先). 成功返回 0. */
int bnn_specmat_mel_basis(int sample_rate, int n_fft, int n_mels,
                          float fmin, float fmax, float *out);

/* 梅尔->线性 增益展开 MelInv[n_bins, n_mels] = Mel^T 按 bin 归一. 成功返回 0. */
int bnn_specmat_mel_inv(const float *mel_basis, int n_mels, int n_bins, float *out);

/* 相位残差 低分辨(P)->线性(n_bins) 线性插值矩阵 PhaseInv[n_bins, P]. 成功返回 0. */
int bnn_specmat_phase_inv(int n_bins, int phase_bands, float *out);

#ifdef __cplusplus
}
#endif
#endif
