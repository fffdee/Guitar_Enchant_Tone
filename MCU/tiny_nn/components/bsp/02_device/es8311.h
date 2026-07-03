/* ES8311 低功耗音频编解码器驱动 (I2C 控制 + I2S 音频)。
 *
 * ES8311 是 Everest Semi 的单声道低功耗 CODEC, 集成 ADC/DAC/耳机放大器。
 * 通过 I2C 配置寄存器, 通过 I2S 传输音频数据。
 * 本驱动基于 esp_codec_dev 组件, 封装了全套初始化/控制/数据流接口。
 *
 * 引脚分配 (用户的 ES8311 板子):
 *   I2C (须单独初始化, 传入 bus_handle):
 *     GPIO8  SDA  (典型)
 *     GPIO9  SCL  (典型)
 *   I2S:
 *     GPIO2  MCLK (主时钟, ES8311 从 MCLK 生成内部时钟)
 *     GPIO3  BCLK (位时钟)
 *     GPIO4  WS   (字选择/LRCK)
 *     GPIO5  DOUT (MCU I2S TX -> ES8311 DAC)
 *     GPIO6  DIN  (ES8311 ADC -> MCU I2S RX)
 *
 * ES8311 I2C 地址: 0x18 (7-bit, AD0=0) 或 0x19 (AD0=1)
 *
 * 用法:
 *   1. 初始化 I2C 总线 (i2c_master_bus_handle_t)
 *   2. es8311_init(bus, sr, MCLK, BCLK, WS, DOUT, DIN, PA, addr)
 *   3. es8311_start()  启用 ADC/DAC
 *   4. es8311_read() / es8311_write() 收发音频
 *   5. es8311_stop() / es8311_deinit() 释放
 */
#ifndef ES8311_H
#define ES8311_H

#include "esp_err.h"
#include <stdint.h>
#include "driver/i2c_master.h"
#include "driver/gpio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ES8311 I2C 引脚 (独立于 I2S 引脚, 用于寄存器配置) */
#define ES8311_I2C_PORT      1        /* I2C 端口号 (wks-p4-cb: NUM_1) */
#define ES8311_I2C_SDA       28       /* SDA 引脚 */
#define ES8311_I2C_SCL       29       /* SCL 引脚 */
#define ES8311_I2C_FREQ      400000   /* 400kHz Fast-mode */
#define ES8311_I2C_ADDR      0x18     /* 7-bit 地址 (AD0 接地) */

/* ES8311 I2S 引脚 (音频数据, wks-p4-cb 验证过的分配) */
#define ES8311_I2S_MCLK      2
#define ES8311_I2S_BCLK      4
#define ES8311_I2S_WS        6
#define ES8311_I2S_DOUT      3
#define ES8311_I2S_DIN       5

/* PA 功放使能引脚 (wks-p4-cb: GPIO11) */
#define ES8311_PA_PIN        11

/* 默认采样率 */
#define ES8311_DEFAULT_SAMPLE_RATE  48000

/* ── 公共 API ── */

/* 初始化 ES8311 并创建 I2S 双工通道。
 * i2c_bus:    已初始化的 I2C 主控总线句柄 (传 NULL 则内部自动初始化 I2C)
 * sample_rate: 采样率 (8000~96000), 传 0 则使用 48000
 * mclk~din:   I2S 引脚, 传 GPIO_NUM_NC 使用默认值
 * pa_pin:      功放使能引脚 (PA), 不需要则传 GPIO_NUM_NC
 * es8311_addr: I2C 设备地址, 传 0 则使用 0x18 */
esp_err_t es8311_init(i2c_master_bus_handle_t i2c_bus, int sample_rate,
                      gpio_num_t mclk, gpio_num_t bclk, gpio_num_t ws,
                      gpio_num_t dout, gpio_num_t din,
                      gpio_num_t pa_pin, uint8_t es8311_addr);

/* 启动 ES8311 (启用输入/输出, 打开 PA) */
esp_err_t es8311_start(void);

/* 停止 ES8311 (禁用输入/输出, 关闭 PA) */
esp_err_t es8311_stop(void);

/* 反初始化, 释放所有资源 (I2C + I2S + codec) */
esp_err_t es8311_deinit(void);

/* 设置输出音量 (0-100, 默认 70) */
esp_err_t es8311_set_volume(int vol);

/* 设置麦克风输入增益 (dB, 典型值 0~42, 默认 24.0) */
esp_err_t es8311_set_mic_gain(float gain_db);

/* 从 I2S RX 读取采样 (16-bit signed, 单声道)。
 * 返回实际读取的采样数, 出错返回 -1。 */
int es8311_read(int16_t *samples, int max_samples, int timeout_ms);

/* 向 I2S TX 写入采样 (16-bit signed, 单声道)。
 * 返回实际写入的采样数, 出错返回 -1。 */
int es8311_write(const int16_t *samples, int num_samples, int timeout_ms);

/* ES8311 是否已初始化 */
int es8311_is_ready(void);

#ifdef __cplusplus
}
#endif
#endif
