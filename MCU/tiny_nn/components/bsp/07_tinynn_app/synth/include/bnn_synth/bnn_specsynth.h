#ifndef BNN_SPECSYNTH_H
#define BNN_SPECSYNTH_H

#include "bnn_frontend/bnn_mask_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 谱掩码声码器 (与 PC 端 mask/reconstruct.py 数值对齐) —— 流式逐帧:
 *   gain_lin = MelInv·mask;  Ylin = gain·mag;  ph = phase + PhaseInv·dphi
 *   noise_shape = NoiseFB^T·noise (随机相位)
 *   谱 = Ylin·e^{j·ph} + noise_shape·e^{j·rp}  -> irfft -> 加窗 OLA -> 每帧出 hop 样点
 * 频谱逆变换委托 op 层 bnn_dsp 后端 (可换 ESP-DSP), 与硬件解耦.
 * 状态: OLA 累加器 + 增益帧间平滑 (a, 默认 0.5) + 噪声 RNG.
 */
typedef struct bnn_specsynth bnn_specsynth_t;

bnn_specsynth_t *bnn_specsynth_create(const bnn_mask_cfg_t *cfg);
void             bnn_specsynth_destroy(bnn_specsynth_t *s);
void             bnn_specsynth_reset(bnn_specsynth_t *s);
void             bnn_specsynth_set_seed(bnn_specsynth_t *s, unsigned int seed);
void             bnn_specsynth_set_smooth(bnn_specsynth_t *s, float a);

/*
 * 处理一帧: mag/phase[n_bins], mask[n_mels], dphi[phase_bands], noise[noise_bands].
 * 写出 hop 个输出样点到 out_hop. add_noise=0 时跳过噪声路径. 成功返回 0.
 */
int bnn_specsynth_process(bnn_specsynth_t *s, const float *mag, const float *phase,
                          const float *mask, const float *dphi, const float *noise,
                          int add_noise, float *out_hop);

/* 诊断: 由 mask 计算线性域 gain_lin[0..n-1] (不含帧间平滑) */
void bnn_specsynth_peek_gain_lin(const bnn_specsynth_t *s, const float *mask,
                                 float *gain_lin, int n);

#ifdef __cplusplus
}
#endif
#endif
