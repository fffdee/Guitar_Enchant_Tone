/* I2S 实时推理流水线: I2S RX -> 推理 -> I2S TX。
 *
 * 工作原理:
 *   1. I2S RX 采集吉他模拟音频 (48kHz/16bit 单声道)
 *   2. 累积到一帧 (hop_size 采样) 后送入 masknet 推理
 *   3. 推理输出通过 I2S TX 播放
 *
 * 流式推理: bnn_masknet 支持逐帧处理, 每帧 hop_size 采样输入输出,
 * 内部维护状态 (卷积因果填充 / 谐波振荡器相位) 保证帧间连续。
 *
 * 双核架构:
 *   core1: I2S 收发 + 推理 (高优先级, 保证实时性)
 *   core0: CLI 命令行 (可随时切换乐器/停止)
 */
#ifndef I2S_XFORM_H
#define I2S_XFORM_H

#include "esp_err.h"
#include "audio_post.h"

#ifdef __cplusplus
extern "C" {
#endif

/* I2S 实时推理配置 */
typedef struct {
    char              instrument[24];   /* 乐器名称 */
    float             pitch_semitones;  /* 变调半音 */
    float             gain_db;          /* 输出增益 dB */
    audio_clip_mode_t clip_mode;        /* 削波模式 */
    int               add_noise;        /* 噪声注入 */
} i2s_xform_cfg_t;

/* 启动 I2S 实时推理 (创建 core1 任务)。
 * 前提: audio_xform_init() 已成功加载模型, i2s_driver_init() 已完成。
 * cfg: 推理配置 (乐器/后处理参数), 不可为 NULL。 */
esp_err_t i2s_xform_start(const i2s_xform_cfg_t *cfg);

/* 停止 I2S 实时推理 (停止 I2S 收发, 删除任务) */
esp_err_t i2s_xform_stop(void);

/* 是否正在运行 */
int i2s_xform_running(void);

/* 运行时切换乐器 (无需停止重启) */
esp_err_t i2s_xform_set_instrument(const char *instrument);

/* 运行时更新后处理参数 */
esp_err_t i2s_xform_set_post(float pitch_semitones, float gain_db,
                              audio_clip_mode_t clip_mode, int add_noise);

/* 获取运行状态信息 (写入 buf) */
void i2s_xform_status(char *buf, size_t n);

#ifdef __cplusplus
}
#endif
#endif
