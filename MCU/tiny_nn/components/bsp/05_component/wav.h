/* WAV 读写 (应用层组件, 不进推理内核). 统一以 float[-1,1] 单声道交互. */
#ifndef WAV_H
#define WAV_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 读取 WAV -> 单声道 float[-1,1]. 支持 16/24/32-bit PCM, 多声道下混.
 * *out 由本函数分配 (优先 PSRAM), 调用方用 free() 释放. 成功返回 ESP_OK. */
esp_err_t wav_read_mono_f32(const char *path, float **out, int *n_samples, int *sample_rate);

/* 写出单声道 float[-1,1] 为 16-bit PCM WAV. 自动裁剪到 [-1,1]. */
esp_err_t wav_write_mono_f32(const char *path, const float *data, int n_samples, int sample_rate);

#ifdef __cplusplus
}
#endif
#endif
