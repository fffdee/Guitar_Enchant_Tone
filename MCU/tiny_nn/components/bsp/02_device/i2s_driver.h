/* I2S STD 双工驱动: 48kHz/16bit 单声道, GPIO2-6。
 * 用于实时吉他音色转换: I2S RX 采集模拟音频 -> 推理 -> I2S TX 输出。
 *
 * 引脚分配 (ESP32-P4):
 *   GPIO2  MCLK  (可选, 部分编解码器需要)
 *   GPIO3  BCLK
 *   GPIO4  WS
 *   GPIO5  DOUT  (MCU -> DAC/耳机)
 *   GPIO6  DIN   (ADC/吉他 -> MCU)
 */
#ifndef I2S_DRIVER_H
#define I2S_DRIVER_H

#include "esp_err.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* I2S 引脚定义 */
#define I2S_MCLK   2
#define I2S_BCLK   3
#define I2S_WS     4
#define I2S_DOUT   5
#define I2S_DIN    6

/* 默认采样率 (与模型一致) */
#define I2S_DEFAULT_SAMPLE_RATE  48000

/* 初始化 I2S 双工通道 (TX + RX 共享 BCLK/WS)。
 * sample_rate: 采样率, 传 0 则使用默认 48000。
 * 成功后通道处于 disable 状态, 需调用 i2s_driver_start() 启动。 */
esp_err_t i2s_driver_init(int sample_rate);

/* 启动 I2S 收发 (enable 两个通道) */
esp_err_t i2s_driver_start(void);

/* 停止 I2S 收发 (disable 两个通道) */
esp_err_t i2s_driver_stop(void);

/* 反初始化, 释放所有资源 */
esp_err_t i2s_driver_deinit(void);

/* 从 I2S RX 读取采样 (16-bit signed, 单声道)。
 * 返回实际读取的采样数, 出错返回 -1。 */
int i2s_driver_read(int16_t *samples, int max_samples, int timeout_ms);

/* 向 I2S TX 写入采样 (16-bit signed, 单声道)。
 * 返回实际写入的采样数, 出错返回 -1。 */
int i2s_driver_write(const int16_t *samples, int num_samples, int timeout_ms);

/* I2S 是否已初始化 */
int i2s_driver_is_ready(void);

#ifdef __cplusplus
}
#endif
#endif
