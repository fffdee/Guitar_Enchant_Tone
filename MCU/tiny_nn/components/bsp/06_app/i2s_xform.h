/* I2S 实时推理流水线: I2S RX -> [音色转换] -> [效果器链] -> I2S TX。
 *
 * 工作原理:
 *   1. I2S RX 采集吉他模拟音频 (48kHz/16bit 单声道)
 *   2. 累积到一帧 (hop_size 采样) 后送入 masknet 推理 (可旁通)
 *   3. 推理输出 (或旁通的原信号) 送入效果器链做后处理 (可旁通)
 *   4. 通过 I2S TX 播放
 *
 * 旁通:
 *   - 音色转换旁通 (bypass=1): 跳过神经网络推理, 输入直通到效果器链
 *   - 效果器链旁通 (fx_chain_set_bypass): 跳过所有效果器
 *   两者可独立旁通, 也可同时旁通 (纯直通)
 *
 * 流式推理: bnn_masknet 支持逐帧处理, 每帧 hop_size 采样输入输出,
 * 内部维护状态 (卷积因果填充 / 谐波振荡器相位) 保证帧间连续。
 *
 * 双核架构:
 *   core1: I2S 收发 + 推理 + 效果器 (高优先级, 保证实时性)
 *   core0: CLI 命令行 (可随时切换乐器/旁通/调参)
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
    int               use_int8;         /* INT8 conv 加速 */
    int               bypass;           /* 音色转换旁通 (1=旁通, 0=正常推理) */
    int               post_fx;          /* 启用效果器链后处理 (1=启用, 0=不启用) */
} i2s_xform_cfg_t;

/* 启动 I2S 实时推理 (创建 core1 任务)。
 * 前提: audio_xform_init() 已成功加载模型 (除非 bypass=1), i2s_driver_init() 已完成。
 * cfg: 推理配置 (乐器/后处理/旁通参数), 不可为 NULL。 */
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

/* 运行时切换音色转换旁通 (1=旁通, 0=正常推理) */
esp_err_t i2s_xform_set_bypass(int bypass);

/* 运行时切换效果器链后处理 (1=启用, 0=不启用) */
esp_err_t i2s_xform_set_post_fx(int enabled);

/* 获取运行状态信息 (写入 buf) */
void i2s_xform_status(char *buf, size_t n);

#ifdef __cplusplus
}
#endif
#endif
