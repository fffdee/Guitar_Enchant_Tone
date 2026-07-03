/* 电吉他效果器链框架。
 *
 * 两种工作方式:
 *   A. 独立模式 (fx_chain_start): 自己管理 I2S 收发, 纯效果器链处理
 *      I2S RX -> 效果器链 -> I2S TX
 *
 *   B. 串联模式 (fx_chain_process): 由 i2s_xform 调用, 作为音色转换的后处理
 *      I2S RX -> 音色转换 -> [效果器链] -> I2S TX
 *      此时 fx_chain 不启动自己的 I2S 任务, 仅提供缓冲处理函数。
 *
 * 旁通:
 *   - 全局旁通 (fx_chain_set_bypass): 整个链跳过, 输出=输入
 *   - 单节点旁通 (fx_chain_set_enabled): 单个效果器跳过
 *
 * 与音色转换 (i2s_xform) 的关系:
 *   - 串联模式: i2s_xform 启动时调用 fx_chain_process, 两者协同
 *   - 独立模式: fx_chain_start 与 i2s_xform 互斥 (共用 I2S)
 */
#ifndef FX_CHAIN_H
#define FX_CHAIN_H

#include "esp_err.h"
#include "fx_modules.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FX_CHAIN_MAX   8   /* 链中最多效果器数量 */
#define FX_HOP_DEFAULT 240 /* 默认帧长 (5ms @48kHz, 与 i2s_xform 一致) */

/* 效果器链配置 (独立模式使用) */
typedef struct {
    int   sample_rate;     /* 采样率 (默认 48000) */
    int   hop_size;        /* 每帧采样数 (默认 240) */
    float input_gain_db;   /* 输入增益 dB */
    float output_gain_db;  /* 输出增益 dB */
} fx_chain_cfg_t;

/* ---------- 独立模式: 自管理 I2S 实时处理 ---------- */

/* 启动效果器链实时处理 (创建 core1 任务)。
 * 前提: i2s_driver_init() 已完成, i2s_xform 未运行。
 * cfg: 链配置, 不可为 NULL。 */
esp_err_t fx_chain_start(const fx_chain_cfg_t *cfg);

/* 停止效果器链 */
esp_err_t fx_chain_stop(void);

/* 是否正在运行 (独立模式) */
int fx_chain_running(void);

/* ---------- 串联模式: 缓冲处理 (供 i2s_xform 调用) ---------- */

/* 处理一段音频缓冲 (in-place)。
 * 用于串联模式: i2s_xform 在推理后调用此函数做后处理。
 * 若全局旁通开启, 直接返回不处理。
 * buf: 音频采样 (float, [-1,1]), 长度 n
 * 返回 ESP_OK。 */
esp_err_t fx_chain_process(float *buf, int n);

/* ---------- 旁通控制 ---------- */

/* 设置全局旁通 (1=旁通整个链, 0=启用) */
esp_err_t fx_chain_set_bypass(int bypass);

/* 获取全局旁通状态 */
int fx_chain_get_bypass(void);

/* ---------- 运行时链操作 (线程安全, 两种模式都可用) ---------- */

/* 在链尾追加效果器, 返回索引 (>=0) 或负错误码 */
int fx_chain_add(fx_type_t type);

/* 在指定位置插入效果器, 返回索引或负错误码 */
int fx_chain_insert(int index, fx_type_t type);

/* 删除指定索引的效果器 */
esp_err_t fx_chain_remove(int index);

/* 清空效果器链 */
esp_err_t fx_chain_clear(void);

/* 启用/旁通指定效果器 (enabled=1 启用, 0 旁通) */
esp_err_t fx_chain_set_enabled(int index, int enabled);

/* 设置指定效果器的参数 (按参数名) */
esp_err_t fx_chain_set_param(int index, const char *param, float value);

/* 重置所有效果器内部状态 (清缓冲等) */
esp_err_t fx_chain_reset(void);

/* 加载预设链 (清空当前链, 加载一组常用效果器) */
esp_err_t fx_chain_load_preset(const char *name);

/* 列出可用预设名 */
void fx_chain_list_presets(char *buf, size_t n);

/* 获取链状态信息 (写入 buf) */
void fx_chain_status(char *buf, size_t n);

/* 打印当前效果器链 (用于 CLI) */
void fx_chain_print(void);

#ifdef __cplusplus
}
#endif
#endif
