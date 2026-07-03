#ifndef BNN_BANDS_H
#define BNN_BANDS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 噪声子带边界 (Mel 间隔) —— 与 PC 端 ddsp/bands.py: make_band_edges 数值一致.
 *
 * 返回 n_bands+1 个 "bin 索引" (基于 fft_size 的 rfft 频格), 用法与 Python 相同:
 *   某子带在合成时的 Hz 边界 = bin_edge * sample_rate / fft_size.
 *
 * 用 HTK 公式 hz<->mel, 在 [fmin, fmax] 上等 mel 取 n_bands+1 个边界, 换算到 bin 并去重单调化.
 *
 *  sample_rate : 采样率 (Hz)
 *  fft_size    : 分析 FFT 点数 (与训练侧一致, 默认 2048)
 *  n_bands     : 噪声子带数 (默认 10)
 *  fmin        : 子带下限 Hz (Python 默认 40.0)
 *  fmax        : 子带上限 Hz, <=0 时取 sample_rate/2
 *  edges_out   : 输出缓冲, 长度 >= n_bands+1
 *
 * 返回写入的边界数 (n_bands+1), 失败返回 <0.
 */
int bnn_bands_make(int sample_rate, int fft_size, int n_bands,
                   float fmin, float fmax, int *edges_out);

#ifdef __cplusplus
}
#endif
#endif
