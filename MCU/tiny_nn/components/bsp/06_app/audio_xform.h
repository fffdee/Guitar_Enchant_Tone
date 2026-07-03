/* 音色转换应用: 读 wav -> bnn_masknet 推理 -> 写 wav. 仅依赖 Tinynn 公共 API, 与 BSP 解耦. */
#ifndef AUDIO_XFORM_H
#define AUDIO_XFORM_H

#include "esp_err.h"
#include "audio_post.h"
#include "model_store.h"
#include "bnn_model/bnn_masknet.h"
#include "bnn_frontend/bnn_mask_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 转换后处理选项 (变调 + 增益 + 削波), 对应 PC 端 render_instrument 的运行时控制 */
typedef struct {
    float             pitch_semitones;  /* 变调半音, 0 = 不变调 */
    float             gain_db;          /* 输出增益 dB, 0 = 不变 */
    audio_clip_mode_t clip_mode;        /* 末级削波: LIMIT/SOFT/HARD */
    int               add_noise;        /* 1=开启频谱噪声注入(防乐音噪声), 0=关(默认, 输出更干净) */
    int               use_int8;         /* 1=启用 INT8 conv 加速 (需 bin 含 weights_i8) */
} audio_xform_opt_t;

/* 加载模型包并构建 masknet. model_path 例: "/sdcard/model/xform_model.bin" */
esp_err_t audio_xform_init(const char *model_path);

/* 模型是否已成功加载 */
int audio_xform_loaded(void);

/* INT8 权重是否已注入且可用 (infer/live -8 前提) */
int audio_xform_int8_ready(void);

/* 打印模型信息 (采样率 / 参数量 / 乐器列表) 到日志, 供 CLI `model` 命令使用 */
void audio_xform_print_model(void);

/* 读 in_wav -> 转换为 instrument 音色 (+可选后处理) -> 写 out_wav.
 * opt 可为 NULL (无后处理). instrument 不在模型中时返回 ESP_ERR_NOT_FOUND 并跳过(不回退). */
esp_err_t audio_xform_file(const char *in_wav, const char *out_wav,
                           const char *instrument, const audio_xform_opt_t *opt);

/*
 * 打印跨多次调用的推理时间统计:
 *   次数 / avg / min / max 耗时 ms / 平均实时比 / workspace 峰值 KB
 */
void audio_xform_print_stats(void);

/* 清零统计累加器 (换模型或基准测试前调用) */
void audio_xform_reset_stats(void);

/* 推理进行中查询进度 (frame/total 帧, pct 0-100); 空闲时 pct=0 */
void audio_xform_infer_progress(int *pct, int *frame, int *total);

/* 诊断: 对短 wav 跑推理并打印首块首帧中间值 (需模型已加载) */
esp_err_t audio_xform_debug(const char *in_wav, const char *instrument);

void audio_xform_deinit(void);

/* 全局变量声明 (供 i2s_xform.c 使用) */
extern model_pkg_t    s_pkg;
extern bnn_masknet_t *s_net;
extern bnn_mask_cfg_t s_cfg;

#ifdef __cplusplus
}
#endif
#endif
