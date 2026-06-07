/* 音色转换应用: 读 wav -> bnn_masknet 推理 -> 写 wav. 仅依赖 Tinynn 公共 API, 与 BSP 解耦. */
#ifndef AUDIO_XFORM_H
#define AUDIO_XFORM_H

#include "esp_err.h"
#include "audio_post.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 转换后处理选项 (变调 + 增益 + 削波), 对应 PC 端 render_instrument 的运行时控制 */
typedef struct {
    float             pitch_semitones;  /* 变调半音, 0 = 不变调 */
    float             gain_db;          /* 输出增益 dB, 0 = 不变 */
    audio_clip_mode_t clip_mode;        /* 末级削波: LIMIT/SOFT/HARD */
} audio_xform_opt_t;

/* 加载模型包并构建 masknet. model_path 例: "/sdcard/model/xform_model.bin" */
esp_err_t audio_xform_init(const char *model_path);

/* 模型是否已成功加载 */
int audio_xform_loaded(void);

/* 打印模型信息 (采样率 / 参数量 / 乐器列表) 到日志, 供 CLI `model` 命令使用 */
void audio_xform_print_model(void);

/* 读 in_wav -> 转换为 instrument 音色 (+可选后处理) -> 写 out_wav.
 * opt 可为 NULL (无后处理). instrument 不在模型中时返回 ESP_ERR_NOT_FOUND 并跳过(不回退). */
esp_err_t audio_xform_file(const char *in_wav, const char *out_wav,
                           const char *instrument, const audio_xform_opt_t *opt);

void audio_xform_deinit(void);

#ifdef __cplusplus
}
#endif
#endif
