/* 音频后处理: 变调(相位声码器) + 增益/削波. 对应 PC 端 mask/audio.py 的
 * pitch_shift_semitones / apply_gain_db, 用于重建链之后的运行时控制. */
#ifndef AUDIO_POST_H
#define AUDIO_POST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AUDIO_CLIP_LIMIT = 0,  /* 干净限幅: 峰值超阈值整体回缩, 不失真 */
    AUDIO_CLIP_SOFT  = 1,  /* 软限幅: tanh 软饱和 (加谐波, 低频更厚) */
    AUDIO_CLIP_HARD  = 2,  /* 硬削波: clamp 到 [-1,1] */
} audio_clip_mode_t;

/* 相位声码器变调 (保持时长). y[n] 原地修改, 升/降 semitones 半音.
 * n_fft/hop/sr 取模型配置. 返回 0 成功, 非 0 失败(y 不变). */
int audio_post_pitch_shift(float *y, int n, int sr, float semitones, int n_fft, int hop);

/* 输出增益(dB) + 末级削波. g=10^(dB/20) 后按 mode 处理过冲. y[n] 原地. */
void audio_post_gain_clip(float *y, int n, float gain_db, audio_clip_mode_t mode);

#ifdef __cplusplus
}
#endif
#endif
