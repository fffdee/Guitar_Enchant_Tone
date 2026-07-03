/* I2S STD 双工驱动实现: 48kHz/16bit 单声道, GPIO2-6。
 * 参考 ESP-IDF i2s_std 例程, 适配 ESP32-P4 音色转换场景。
 *
 * 数据流: ADC/DIN -> I2S RX -> 推理 -> I2S TX -> DAC/DOUT
 * 双工模式: TX 和 RX 共享 BCLK/WS, 同一 I2S 控制器。
 */
#include "i2s_driver.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include <stdlib.h>

static const char *TAG = "i2s_drv";

static i2s_chan_handle_t s_tx_chan = NULL;
static i2s_chan_handle_t s_rx_chan = NULL;
static int s_sample_rate = I2S_DEFAULT_SAMPLE_RATE;
static int s_ready = 0;

/* 可配置引脚 (默认值可在 init 前通过 i2s_driver_set_pins 修改) */
static struct {
    gpio_num_t mclk;
    gpio_num_t bclk;
    gpio_num_t ws;
    gpio_num_t dout;
    gpio_num_t din;
    int        pins_set;   /* 是否已通过 i2s_driver_set_pins 设置 */
} s_pins = { I2S_MCLK, I2S_BCLK, I2S_WS, I2S_DOUT, I2S_DIN, 0 };

void i2s_driver_set_pins(gpio_num_t mclk, gpio_num_t bclk, gpio_num_t ws,
                          gpio_num_t dout, gpio_num_t din)
{
    s_pins.mclk = mclk;
    s_pins.bclk = bclk;
    s_pins.ws   = ws;
    s_pins.dout = dout;
    s_pins.din  = din;
    s_pins.pins_set = 1;
    ESP_LOGI(TAG, "引脚覆盖: MCLK=%d BCLK=%d WS=%d DOUT=%d DIN=%d",
             (int)mclk, (int)bclk, (int)ws, (int)dout, (int)din);
}

esp_err_t i2s_driver_init(int sample_rate)
{
    if (s_ready) return ESP_OK;

    s_sample_rate = (sample_rate > 0) ? sample_rate : I2S_DEFAULT_SAMPLE_RATE;

    /* Step 1: 分配 I2S 双工通道 (TX + RX 共享 BCLK/WS) */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    /* DMA 缓冲: 4×240×4B=3840B=20ms @48kHz stereo (平衡延迟与抗欠载) */
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 240;
    esp_err_t err = i2s_new_channel(&chan_cfg, &s_tx_chan, &s_rx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel 失败: %s", esp_err_to_name(err));
        return err;
    }

    /* Step 2: 配置 STD 模式 — 48kHz/16bit/立体声/Philips 格式
     *
     * ★ 必须用 STEREO! WM8978 R4 配置为立体声 (MONO=0),
     *   输出 L+R 两个 slot 的数据。用 MONO 模式会导致 BCLK/WS 时序
     *   不匹配, ADC 数据无法被正确读取 → 录音静音。
     *   参考 BH-F407 工程同样使用 STEREO 模式。 */
    i2s_std_config_t std_cfg = {
        .clk_cfg = {
            .sample_rate_hz = (uint32_t)s_sample_rate,
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .mclk_multiple = I2S_MCLK_MULTIPLE_256,
        },
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = s_pins.mclk,
            .bclk = s_pins.bclk,
            .ws   = s_pins.ws,
            .dout = s_pins.dout,
            .din  = s_pins.din,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    /* ★ 显式设置 slot_bit_width 和 ws_width, 避免默认行为差异 */
    std_cfg.slot_cfg.slot_bit_width = I2S_DATA_BIT_WIDTH_16BIT;
    std_cfg.slot_cfg.ws_width       = I2S_DATA_BIT_WIDTH_16BIT;

    /* TX: SLOT_BOTH + 立体声交织数据。
     *     i2s_driver_write() 内部将 mono 数据展开为 L=R 的立体声对,
     *     确保总线时序完全匹配 WM8978 的 STEREO 格式。
     *
     *     ★ STEREO + SLOT_BOTH 是唯一与 WM8978 R4 配置完全一致的方案。
     *       使用 SLOT_LEFT 在 ESP-IDF v5.x 中可能导致 TX DMA 时序混乱,
     *       数据无法被 WM8978 正确接收 → 输出静音。 */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;

    /* TX (Master): 配置完整的时钟和引脚 */
    err = i2s_channel_init_std_mode(s_tx_chan, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TX init_std_mode 失败: %s", esp_err_to_name(err));
        i2s_del_channel(s_tx_chan); s_tx_chan = NULL;
        i2s_del_channel(s_rx_chan); s_rx_chan = NULL;
        return err;
    }

    /* RX (Slave): 全双工模式下 BCLK/WS 由 TX 提供。
     *
     * ★★ 关键! ESP-IDF v5.x 中, 若 RX 的 BCLK/WS 设为 GPIO_NUM_NC,
     *    RX DMA 可能无法正确绑定共享时钟 → 收不到任何 I2S 数据!
     *    TX 已配置并持有 BCLK/WS 引脚, RX 必须设置相同的引脚号,
     *    让驱动知道 RX 应监听这些引脚的时钟信号。
     *
     * ★ STEREO + SLOT_LEFT: WM8978 输出 L+R 两个 slot,
     *   只取 L slot 的数据。i2s_channel_read() 返回提取后的 mono 数据。 */
    i2s_std_config_t rx_std_cfg = std_cfg;  /* 拷贝 TX 配置 */
    rx_std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  /* 只取左声道 */
    rx_std_cfg.gpio_cfg.mclk = GPIO_NUM_NC;       /* RX 不需要 MCLK */
    /* ★ BCLK/WS 必须设为与 TX 相同的引脚 (告诉驱动 RX 应监听这些时钟) */
    rx_std_cfg.gpio_cfg.bclk = s_pins.bclk;
    rx_std_cfg.gpio_cfg.ws   = s_pins.ws;
    rx_std_cfg.gpio_cfg.dout = GPIO_NUM_NC;        /* RX 不需要 DOUT */
    /* .din 保持原值 (s_pins.din) — RX 唯一的数据输入引脚 */

    err = i2s_channel_init_std_mode(s_rx_chan, &rx_std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "RX init_std_mode 失败: %s", esp_err_to_name(err));
        i2s_del_channel(s_tx_chan); s_tx_chan = NULL;
        i2s_del_channel(s_rx_chan); s_rx_chan = NULL;
        return err;
    }

    s_ready = 1;
    ESP_LOGI(TAG, "I2S 双工初始化完成: %dHz 16bit stereo  MCLK=%d BCLK=%d WS=%d DOUT=%d DIN=%d",
             s_sample_rate, (int)s_pins.mclk, (int)s_pins.bclk, (int)s_pins.ws,
             (int)s_pins.dout, (int)s_pins.din);
    return ESP_OK;
}

esp_err_t i2s_driver_start(void)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    esp_err_t err;
    /* 先启动 TX (输出静音), 再启动 RX, 避免时钟不同步 */
    err = i2s_channel_enable(s_tx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TX enable 失败: %s", esp_err_to_name(err));
        return err;
    }
    err = i2s_channel_enable(s_rx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "RX enable 失败: %s", esp_err_to_name(err));
        i2s_channel_disable(s_tx_chan);
        return err;
    }
    ESP_LOGI(TAG, "I2S 收发已启动");
    return ESP_OK;
}

esp_err_t i2s_driver_stop(void)
{
    if (!s_ready) return ESP_ERR_INVALID_STATE;
    /* 忽略 disable 的返回值: 通道可能已被 DMA 完成或错误事件自动停用,
     * 此时 disable 返回 ESP_ERR_INVALID_STATE 是正常的, 无需报错。 */
    if (s_rx_chan) i2s_channel_disable(s_rx_chan);
    if (s_tx_chan) i2s_channel_disable(s_tx_chan);
    ESP_LOGI(TAG, "I2S 收发已停止");
    return ESP_OK;
}

esp_err_t i2s_driver_deinit(void)
{
    if (!s_ready) return ESP_OK;
    i2s_driver_stop();
    if (s_tx_chan) { i2s_del_channel(s_tx_chan); s_tx_chan = NULL; }
    if (s_rx_chan) { i2s_del_channel(s_rx_chan); s_rx_chan = NULL; }

    /* ★ 显式复位 I2S 引脚, 确保下一次 init 无状态残留
     *    (i2s_del_channel 不一定将引脚恢复为 GPIO 默认状态) */
    gpio_reset_pin(s_pins.mclk);
    gpio_reset_pin(s_pins.bclk);
    gpio_reset_pin(s_pins.ws);
    gpio_reset_pin(s_pins.dout);
    gpio_reset_pin(s_pins.din);

    s_ready = 0;
    ESP_LOGI(TAG, "I2S 已释放");
    return ESP_OK;
}

int i2s_driver_read(int16_t *samples, int max_samples, int timeout_ms)
{
    if (!s_ready || !s_rx_chan) return -1;
    size_t r_bytes = 0;
    size_t req = (size_t)max_samples * sizeof(int16_t);
    esp_err_t err = i2s_channel_read(s_rx_chan, samples, req, &r_bytes, pdMS_TO_TICKS(timeout_ms));
    if (err != ESP_OK) {
        /* 限速: 最多每秒打印一次超时错误, 避免刷屏 */
        static int64_t last_log_us = 0;
        int64_t now = esp_timer_get_time();
        if (now - last_log_us > 1000000) {
            ESP_LOGE(TAG, "RX read 失败: %s (req=%d bytes)", esp_err_to_name(err), (int)req);
            last_log_us = now;
        }
        return -1;
    }
    if (r_bytes == 0) {
        /* 超时无数据 — 只打印一次警告, 避免刷屏 */
        static int rx_timeout_warned = 0;
        if (!rx_timeout_warned) {
            ESP_LOGW(TAG, "RX read 超时: 0 字节 (DMA 未收到任何 I2S 数据, 检查 DIN 接线和 WM8978 ADC 配置)");
            rx_timeout_warned = 1;
        }
        return 0;
    }
    return (int)(r_bytes / sizeof(int16_t));
}

int i2s_driver_write(const int16_t *samples, int num_samples, int timeout_ms)
{
    if (!s_ready || !s_tx_chan) return -1;

    /* TX 使用 STEREO + SLOT_BOTH 模式, 必须写入立体声交织数据。
     * 将 mono 数据展开为 L=R 的 stereo 对: [L0,R0, L1,R1, ...] = [s0,s0, s1,s1, ...] */
    int stereo_count = num_samples * 2;
    int16_t *stereo_buf = (int16_t *)malloc((size_t)stereo_count * sizeof(int16_t));
    if (!stereo_buf) return -1;

    for (int i = 0; i < num_samples; i++) {
        stereo_buf[i * 2]     = samples[i];   /* L */
        stereo_buf[i * 2 + 1] = samples[i];   /* R = L (mono→stereo) */
    }

    size_t w_bytes = 0;
    size_t req = (size_t)stereo_count * sizeof(int16_t);
    esp_err_t err = i2s_channel_write(s_tx_chan, stereo_buf, req, &w_bytes, pdMS_TO_TICKS(timeout_ms));
    free(stereo_buf);

    if (err != ESP_OK) return -1;
    /* 返回写入的 mono 采样数 (写入了 stereo_count/2 = num_samples) */
    return (int)(w_bytes / sizeof(int16_t) / 2);
}

int i2s_driver_is_ready(void)
{
    return s_ready;
}
