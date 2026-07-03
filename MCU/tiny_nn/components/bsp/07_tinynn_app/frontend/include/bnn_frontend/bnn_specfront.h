#ifndef BNN_SPECFRONT_H
#define BNN_SPECFRONT_H

#include "bnn_frontend/bnn_mask_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 谱掩码方案前端 (与 PC 端 mask/audio.py 数值对齐):
 *   单帧(n_fft 样点) -> Hann 加窗 -> rfft -> 幅度/相位 -> 对数梅尔(标准化).
 *   幅度类型 magnitude(非功率); log(mel+log_eps); (x-mean)/std.
 * FFT 委托 op 层 bnn_dsp 后端 (可换 ESP-DSP), 与硬件解耦.
 */
typedef struct bnn_specfront bnn_specfront_t;

bnn_specfront_t *bnn_specfront_create(const bnn_mask_cfg_t *cfg);
void             bnn_specfront_destroy(bnn_specfront_t *fe);

/* 设定梅尔标准化 mean/std (各 n_mels). 不调用则不标准化 (mean=0,std=1). */
void bnn_specfront_set_norm(bnn_specfront_t *fe, const float *mel_mean, const float *mel_std);

/* 提取单帧: frame 长度 = n_fft. logmel[n_mels]/mag[n_bins] 任一可为 NULL.
 * phase 若非 NULL, 需 2*n_bins 浮点, 存单位相位 (cos,sin) 交错. */
void bnn_specfront_extract(bnn_specfront_t *fe, const float *frame,
                           float *logmel, float *mag, float *phase);

#ifdef __cplusplus
}
#endif
#endif
