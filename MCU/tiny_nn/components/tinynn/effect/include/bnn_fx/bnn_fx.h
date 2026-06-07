#ifndef BNN_FX_H
#define BNN_FX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 失真效果器 (传统 DSP, 对应 §7) —— 与神经网络音色转换正交, 零 NN 算力.
 *   波形整形(tanh/cubic/hard 软硬削波) + 可选过采样抗混叠(2x/4x).
 *   流式块处理, 内部维持 FIR 延迟线; 数值与 PC 端 src/esp_xform/fx.py 一致.
 */
typedef enum {
    BNN_FX_TANH = 0,   /* y = tanh(drive*x) */
    BNN_FX_CUBIC = 1,  /* 立方软削波 */
    BNN_FX_HARD = 2    /* 硬削波 clamp */
} bnn_fx_shaper_t;

typedef struct {
    bnn_fx_shaper_t shaper;
    float drive;       /* 驱动量 */
    float out_level;   /* 输出电平 */
    float mix;         /* 干湿比 0..1 (0=干, 1=全湿) */
    int   oversample;  /* 1/2/4 */
} bnn_fx_cfg_t;

static inline void bnn_fx_cfg_default(bnn_fx_cfg_t *c) {
    c->shaper = BNN_FX_TANH; c->drive = 2.0f; c->out_level = 1.0f; c->mix = 1.0f; c->oversample = 2;
}

typedef struct bnn_fx bnn_fx_t;

bnn_fx_t *bnn_fx_create(const bnn_fx_cfg_t *cfg);
void      bnn_fx_destroy(bnn_fx_t *fx);
void      bnn_fx_reset(bnn_fx_t *fx);

/* 流式块处理: in[n] -> out[n] (可原地, out==in 允许). 成功返回 0. */
int       bnn_fx_process(bnn_fx_t *fx, const float *in, int n, float *out);

#ifdef __cplusplus
}
#endif
#endif
