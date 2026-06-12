/* I2S STD 双工驱动实现: 48kHz/16bit 单声道, GPIO2-6。
 * 参考 ESP-IDF i2s_std 例程, 适配 ESP32-P4 音色转换场景。
 *
 * 数据流: ADC/DIN -> I2S RX -> 推理 -> I2S TX -> DAC/DOUT
 * 双工模式: TX 和 RX 共享 BCLK/WS, 同一 I2S 控制器。
 */
#include "i2s_driver.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"

static const char *TAG = "i2s_drv";

static i2s_chan_handle_t s_tx_chan = NULL;
static i2s_chan_handle_t s_rx_chan = NULL;
static int s_sample_rate = I2S_DEFAULT_SAMPLE_RATE;
static int s_ready = 0;

esp_err_t i2s_driver_init(int sample_rate)
{
    if (s_ready) return ESP_OK;

    s_sample_rate = (sample_rate > 0) ? sample_rate : I2S_DEFAULT_SAMPLE_RATE;

    /* Step 1: 分配 I2S 双工通道 (TX + RX 共享 BCLK/WS) */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    /* 增大 DMA 缓冲, 减少欠载/溢出 */
    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = 240;
    esp_err_t err = i2s_new_channel(&chan_cfg, &s_tx_chan, &s_rx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel 失败: %s", esp_err_to_name(err));
        return err;
    }

    /* Step 2: 配置 STD 模式 — 48kHz/16bit/单声道/Philips 格式 */
    i2s_std_config_t std_cfg = {
        .clk_cfg = {
            .sample_rate_hz = (uint32_t)s_sample_rate,
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .mclk_multiple = I2S_MCLK_MULTIPLE_256,
        },
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_MCLK,
            .bclk = I2S_BCLK,
            .ws   = I2S_WS,
            .dout = I2S_DOUT,
            .din  = I2S_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    /* 单声道模式下默认使用左声道, 改为右声道 (多数编解码器右声道对应单声道输入) */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_RIGHT;

    err = i2s_channel_init_std_mode(s_tx_chan, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TX init_std_mode 失败: %s", esp_err_to_name(err));
        i2s_del_channel(s_tx_chan); s_tx_chan = NULL;
        i2s_del_channel(s_rx_chan); s_rx_chan = NULL;
        return err;
    }
    err = i2s_channel_init_std_mode(s_rx_chan, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "RX init_std_mode 失败: %s", esp_err_to_name(err));
        i2s_del_channel(s_tx_chan); s_tx_chan = NULL;
        i2s_del_channel(s_rx_chan); s_rx_chan = NULL;
        return err;
    }

    s_ready = 1;
    ESP_LOGI(TAG, "I2S 双工初始化完成: %dHz 16bit mono  MCLK=%d BCLK=%d WS=%d DOUT=%d DIN=%d",
             s_sample_rate, I2S_MCLK, I2S_BCLK, I2S_WS, I2S_DOUT, I2S_DIN);
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
    i2s_channel_disable(s_rx_chan);
    i2s_channel_disable(s_tx_chan);
    ESP_LOGI(TAG, "I2S 收发已停止");
    return ESP_OK;
}

esp_err_t i2s_driver_deinit(void)
{
    if (!s_ready) return ESP_OK;
    i2s_driver_stop();
    if (s_tx_chan) { i2s_del_channel(s_tx_chan); s_tx_chan = NULL; }
    if (s_rx_chan) { i2s_del_channel(s_rx_chan); s_rx_chan = NULL; }
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
    if (err != ESP_OK) return -1;
    return (int)(r_bytes / sizeof(int16_t));
}

int i2s_driver_write(const int16_t *samples, int num_samples, int timeout_ms)
{
    if (!s_ready || !s_tx_chan) return -1;
    size_t w_bytes = 0;
    size_t req = (size_t)num_samples * sizeof(int16_t);
    esp_err_t err = i2s_channel_write(s_tx_chan, samples, req, &w_bytes, pdMS_TO_TICKS(timeout_ms));
    if (err != ESP_OK) return -1;
    return (int)(w_bytes / sizeof(int16_t));
}

int i2s_driver_is_ready(void)
{
    return s_ready;
}
