#ifndef BNN_XFORM_CONFIG_H
#define BNN_XFORM_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ESP-GT-XFORM 运行期配置 (与 PC 端 configs/train_ddsp.yaml 默认值一致).
 * frontend / synth / runtime 共用. 默认值由 bnn_xform_cfg_default 填充.
 */
typedef struct bnn_xform_cfg {
    int   sample_rate;     /* 48000 */
    int   frame_size;      /* 1536  */
    int   hop_size;        /* 512   */
    int   fft_size;        /* 2048  */

    int   n_mels;          /* 26 */
    int   n_mfcc;          /* 13 */
    int   feature_dim;     /* 20 */
    float fmin;            /* 30   Mel 下限 */
    float fmax;            /* 20000 Mel 上限 */
    float f0_min;          /* 55   */
    float f0_max;          /* 1320 */
    float rolloff_percent; /* 0.85 */
    float silence_db;      /* -60  */

    int   n_harmonics;     /* 30 */
    int   n_noise_bands;   /* 10 */
} bnn_xform_cfg_t;

static inline void bnn_xform_cfg_default(bnn_xform_cfg_t *c) {
    c->sample_rate = 48000; c->frame_size = 1536; c->hop_size = 512; c->fft_size = 2048;
    c->n_mels = 26; c->n_mfcc = 13; c->feature_dim = 20;
    c->fmin = 30.0f; c->fmax = 20000.0f;
    c->f0_min = 55.0f; c->f0_max = 1320.0f;
    c->rolloff_percent = 0.85f; c->silence_db = -60.0f;
    c->n_harmonics = 30; c->n_noise_bands = 10;
}

#define BNN_XFORM_OUTPUT_DIM(c) ((c)->n_harmonics + (c)->n_noise_bands)  /* 40 */
#define BNN_XFORM_INPUT_DIM(c, emb)  ((c)->feature_dim + (emb))          /* 20+8=28 */

#ifdef __cplusplus
}
#endif
#endif
