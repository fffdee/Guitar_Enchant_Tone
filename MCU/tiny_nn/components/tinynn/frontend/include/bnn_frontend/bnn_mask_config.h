#ifndef BNN_MASK_CONFIG_H
#define BNN_MASK_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 频域谱掩码方案运行期配置 (与 PC 端 src/esp_xform/mask/config.py 附录B 默认一致).
 * frontend(specfront) / synth(specsynth) / model(masknet) 共用.
 */
typedef struct bnn_mask_cfg {
    int   sample_rate;   /* 48000 */
    int   n_fft;         /* 1024  */
    int   hop;           /* 256   */
    int   n_bins;        /* 513 = n_fft/2+1 (由 default 填) */

    int   n_mels;        /* 96  */
    float fmin;          /* 40    */
    float fmax;          /* 16000 */
    float log_eps;       /* 1e-5  log(mel+eps) */

    /* 模型结构 */
    int   hidden;        /* 128 */
    int   kernel;        /* 3   */
    int   dilation2;     /* 2   L2 膨胀 */
    int   emb_dim;       /* 16  乐器嵌入 */

    /* 三头 */
    float gmax;          /* 4.0   幅度掩码上限 */
    float dphi_max;      /* π/2   相位残差上限 */
    int   phase_bands;   /* 64  */
    int   noise_bands;   /* 16  */
} bnn_mask_cfg_t;

static inline void bnn_mask_cfg_default(bnn_mask_cfg_t *c) {
    c->sample_rate = 48000; c->n_fft = 1024; c->hop = 256; c->n_bins = 1024 / 2 + 1;
    c->n_mels = 96; c->fmin = 40.0f; c->fmax = 16000.0f; c->log_eps = 1e-5f;
    c->hidden = 128; c->kernel = 3; c->dilation2 = 2; c->emb_dim = 16;
    c->gmax = 4.0f; c->dphi_max = 1.57079632679f; c->phase_bands = 64; c->noise_bands = 16;
}

#define BNN_MASK_COND_DIM(c)  ((c)->emb_dim)   /* 暂仅乐器嵌入; 演奏法嵌入为可选扩展 */

#ifdef __cplusplus
}
#endif
#endif
