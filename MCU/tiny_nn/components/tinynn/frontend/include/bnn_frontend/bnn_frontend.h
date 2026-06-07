#ifndef BNN_FRONTEND_H
#define BNN_FRONTEND_H

#include "bnn_frontend/bnn_xform_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 吉他特征前端 (与 Python GuitarFeatureExtractor 数值对齐):
 *   单帧(frame_size 样点) -> 20 维特征 + 基频 f0 + 有声标志.
 *   内部用 op 层的 FFT 后端 (可被 ESP-DSP 替换), 并维护上一帧幅度谱以计算 spectral_flux,
 *   因此是"流式"的; 切换音轨/重新开始时调用 bnn_frontend_reset.
 *
 * 20 维布局: [log_f0, loudness_db, mfcc0..12, centroid, rolloff, voiced, zcr, flux]
 */
typedef struct bnn_frontend bnn_frontend_t;

bnn_frontend_t *bnn_frontend_create(const bnn_xform_cfg_t *cfg);
void            bnn_frontend_destroy(bnn_frontend_t *fe);
void            bnn_frontend_reset(bnn_frontend_t *fe);

/* 提取单帧特征. frame 长度 = cfg.frame_size. 输出 feat 长度 = cfg.feature_dim.
 * f0_hz / voiced 可为 NULL (合成时需要 f0). */
void bnn_frontend_extract(bnn_frontend_t *fe, const float *frame,
                          float *feat, float *f0_hz, float *voiced);

/* 用导出的 mean/std 对特征做标准化 (in-place). 长度均为 cfg.feature_dim. */
void bnn_frontend_standardize(const bnn_xform_cfg_t *cfg, const float *mean,
                              const float *std, float *feat);

#ifdef __cplusplus
}
#endif
#endif
