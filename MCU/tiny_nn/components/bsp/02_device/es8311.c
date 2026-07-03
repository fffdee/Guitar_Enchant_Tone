/* ES8311 编解码器驱动实现。
 *
 * 基于 esp_codec_dev 组件, 封装了:
 *   1. I2C 总线初始化 (寄存器控制)
 *   2. I2S 双工通道创建 (音频数据传输)
 *   3. ES8311 codec 初始化 (时钟/电源/ADC/DAC/音量)
 *   4. 音频数据读写接口
 *
 * 数据流:
 *   模拟输入 → ES8311 ADC → I2S RX → es8311_read() → 推理 → es8311_write() → I2S TX → ES8311 DAC → 耳机/音箱
 */
#include "es8311.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "es8311";

/* DMA 配置 (与 i2s_driver 保持一致) */
#define ES8311_DMA_DESC_NUM   6
#define ES8311_DMA_FRAME_NUM  240

/* 运行时状态 */
static struct {
    i2c_master_bus_handle_t  i2c_bus;       /* I2C 总线 (内部创建则负责释放) */
    int                      i2c_owned;     /* I2C 总线是否由本驱动创建 */
    i2s_chan_handle_t        tx_chan;       /* I2S TX 通道 */
    i2s_chan_handle_t        rx_chan;       /* I2S RX 通道 */
    const audio_codec_data_if_t *data_if;   /* I2S 数据接口 */
    const audio_codec_ctrl_if_t *ctrl_if;   /* I2C 控制接口 */
    const audio_codec_gpio_if_t *gpio_if;   /* GPIO 控制接口 */
    const audio_codec_if_t  *codec_if;      /* ES8311 codec 接口 */
    esp_codec_dev_handle_t   dev;           /* codec 设备句柄 */
    gpio_num_t               pa_pin;        /* PA 功放引脚 */
    int                      sample_rate;   /* 当前采样率 */
    int                      ready;         /* 是否已初始化 */
    int                      running;       /* 是否已启动 */
    int                      volume;        /* 输出音量 0-100 */
} s_es = {0};

/* ── 内部: I2C 总线初始化 ──────────────────────────────────────────── */

static esp_err_t es8311_i2c_init(void)
{
    if (s_es.i2c_bus) return ESP_OK;

    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = ES8311_I2C_PORT,
        .sda_io_num = ES8311_I2C_SDA,
        .scl_io_num = ES8311_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&bus_cfg, &s_es.i2c_bus);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C 总线初始化失败: %s", esp_err_to_name(err));
        return err;
    }
    s_es.i2c_owned = 1;
    ESP_LOGI(TAG, "I2C 初始化: SDA=%d SCL=%d freq=%dHz",
             ES8311_I2C_SDA, ES8311_I2C_SCL, ES8311_I2C_FREQ);
    return ESP_OK;
}

/* ── 内部: I2S 双工通道创建 ────────────────────────────────────────── */

static esp_err_t es8311_create_i2s_channels(gpio_num_t mclk, gpio_num_t bclk,
                                             gpio_num_t ws, gpio_num_t dout,
                                             gpio_num_t din)
{
    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = ES8311_DMA_DESC_NUM,
        .dma_frame_num = ES8311_DMA_FRAME_NUM,
        .auto_clear_after_cb = true,
        .auto_clear_before_cb = false,
        .intr_priority = 0,
    };
    esp_err_t err = i2s_new_channel(&chan_cfg, &s_es.tx_chan, &s_es.rx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel 失败: %s", esp_err_to_name(err));
        return err;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = {
            .sample_rate_hz = (uint32_t)s_es.sample_rate,
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .mclk_multiple = I2S_MCLK_MULTIPLE_256,
#ifdef I2S_HW_VERSION_2
            .ext_clk_freq_hz = 0,
#endif
        },
        .slot_cfg = {
            .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
            .slot_bit_width = I2S_SLOT_BIT_WIDTH_AUTO,
            .slot_mode = I2S_SLOT_MODE_STEREO,
            .slot_mask = I2S_STD_SLOT_BOTH,
            .ws_width = I2S_DATA_BIT_WIDTH_16BIT,
            .ws_pol = false,
            .bit_shift = true,
#ifdef I2S_HW_VERSION_2
            .left_align = true,
            .big_endian = false,
            .bit_order_lsb = false,
#endif
        },
        .gpio_cfg = {
            .mclk = mclk,
            .bclk = bclk,
            .ws = ws,
            .dout = dout,
            .din = din,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    err = i2s_channel_init_std_mode(s_es.tx_chan, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TX init_std_mode 失败: %s", esp_err_to_name(err));
        return err;
    }
    err = i2s_channel_init_std_mode(s_es.rx_chan, &std_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "RX init_std_mode 失败: %s", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "I2S 双工通道创建: %dHz MCLK=%d BCLK=%d WS=%d DOUT=%d DIN=%d",
             s_es.sample_rate, (int)mclk, (int)bclk, (int)ws, (int)dout, (int)din);
    return ESP_OK;
}

/* ── 公共 API 实现 ────────────────────────────────────────────────── */

esp_err_t es8311_init(i2c_master_bus_handle_t i2c_bus, int sample_rate,
                      gpio_num_t mclk, gpio_num_t bclk, gpio_num_t ws,
                      gpio_num_t dout, gpio_num_t din,
                      gpio_num_t pa_pin, uint8_t es8311_addr)
{
    if (s_es.ready) return ESP_OK;

    s_es.sample_rate = (sample_rate > 0) ? sample_rate : ES8311_DEFAULT_SAMPLE_RATE;
    s_es.volume = 70;           /* 默认 70% 音量 */
    s_es.pa_pin = pa_pin;

    /* 使用默认 GPIO 值 (如果没指定) */
    if (mclk == GPIO_NUM_NC) mclk = ES8311_I2S_MCLK;
    if (bclk == GPIO_NUM_NC) bclk = ES8311_I2S_BCLK;
    if (ws   == GPIO_NUM_NC) ws   = ES8311_I2S_WS;
    if (dout == GPIO_NUM_NC) dout = ES8311_I2S_DOUT;
    if (din  == GPIO_NUM_NC) din  = ES8311_I2S_DIN;
    if (es8311_addr == 0) es8311_addr = ES8311_I2C_ADDR;

    /* Step 1: I2C 总线 */
    if (i2c_bus) {
        s_es.i2c_bus = i2c_bus;
        s_es.i2c_owned = 0;
    } else {
        esp_err_t err = es8311_i2c_init();
        if (err) return err;
    }

    /* Step 2: 创建 I2S 双工通道 */
    esp_err_t err = es8311_create_i2s_channels(mclk, bclk, ws, dout, din);
    if (err) {
        if (s_es.i2c_owned && s_es.i2c_bus) {
            i2c_del_master_bus(s_es.i2c_bus);
            s_es.i2c_bus = NULL; s_es.i2c_owned = 0;
        }
        return err;
    }

    /* Step 3: esp_codec_dev 数据/控制/GPIO 接口 */
    audio_codec_i2s_cfg_t i2s_cfg = {
        .port = I2S_NUM_0,
        .rx_handle = s_es.rx_chan,
        .tx_handle = s_es.tx_chan,
    };
    s_es.data_if = audio_codec_new_i2s_data(&i2s_cfg);
    if (!s_es.data_if) {
        ESP_LOGE(TAG, "audio_codec_new_i2s_data 失败");
        goto fail_i2s;
    }

    audio_codec_i2c_cfg_t i2c_cfg = {
        .port = ES8311_I2C_PORT,
        .addr = es8311_addr,
        .bus_handle = s_es.i2c_bus,
    };
    s_es.ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    if (!s_es.ctrl_if) {
        ESP_LOGE(TAG, "audio_codec_new_i2c_ctrl 失败 (检查 I2C 地址 0x%02X)", es8311_addr);
        goto fail_ctrl;
    }

    s_es.gpio_if = audio_codec_new_gpio();
    if (!s_es.gpio_if) {
        ESP_LOGE(TAG, "audio_codec_new_gpio 失败");
        goto fail_gpio;
    }

    /* Step 4: 创建 ES8311 codec 接口 */
    es8311_codec_cfg_t es8311_cfg = {
        .ctrl_if = s_es.ctrl_if,
        .gpio_if = s_es.gpio_if,
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_BOTH,
        .pa_pin = pa_pin,
        .use_mclk = true,
        .hw_gain = {
            .pa_voltage = 5.0,
            .codec_dac_voltage = 3.3,
        },
        .pa_reverted = false,
    };
    s_es.codec_if = es8311_codec_new(&es8311_cfg);
    if (!s_es.codec_if) {
        ESP_LOGE(TAG, "es8311_codec_new 失败 (I2C 设备无响应?)");
        goto fail_codec;
    }

    /* Step 5: 创建并打开 codec 设备 */
    esp_codec_dev_cfg_t dev_cfg = {
        .dev_type = ESP_CODEC_DEV_TYPE_IN_OUT,
        .codec_if = s_es.codec_if,
        .data_if = s_es.data_if,
    };
    s_es.dev = esp_codec_dev_new(&dev_cfg);
    if (!s_es.dev) {
        ESP_LOGE(TAG, "esp_codec_dev_new 失败");
        goto fail_dev;
    }

    esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = 16,
        .channel = 1,
        .channel_mask = 0,
        .sample_rate = (uint32_t)s_es.sample_rate,
        .mclk_multiple = 0,
    };
    err = esp_codec_dev_open(s_es.dev, &fs);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_codec_dev_open 失败: %s", esp_err_to_name(err));
        goto fail_open;
    }

    /* Step 6: 默认配置 */
    esp_codec_dev_set_in_gain(s_es.dev, 24.0f);   /* 麦克风增益 24dB */
    esp_codec_dev_set_out_vol(s_es.dev, s_es.volume);

    s_es.ready = 1;
    ESP_LOGI(TAG, "ES8311 初始化完成: %dHz 16bit mono  addr=0x%02X",
             s_es.sample_rate, es8311_addr);
    return ESP_OK;

    /* 错误回滚 */
fail_open:
    esp_codec_dev_delete(s_es.dev); s_es.dev = NULL;
fail_dev:
    audio_codec_delete_codec_if(s_es.codec_if); s_es.codec_if = NULL;
fail_codec:
    audio_codec_delete_gpio_if(s_es.gpio_if); s_es.gpio_if = NULL;
fail_gpio:
    audio_codec_delete_ctrl_if(s_es.ctrl_if); s_es.ctrl_if = NULL;
fail_ctrl:
    audio_codec_delete_data_if(s_es.data_if); s_es.data_if = NULL;
fail_i2s:
    if (s_es.tx_chan) { i2s_del_channel(s_es.tx_chan); s_es.tx_chan = NULL; }
    if (s_es.rx_chan) { i2s_del_channel(s_es.rx_chan); s_es.rx_chan = NULL; }
    if (s_es.i2c_owned && s_es.i2c_bus) {
        i2c_del_master_bus(s_es.i2c_bus);
        s_es.i2c_bus = NULL; s_es.i2c_owned = 0;
    }
    return ESP_FAIL;
}

esp_err_t es8311_start(void)
{
    if (!s_es.ready) return ESP_ERR_INVALID_STATE;
    if (s_es.running) return ESP_OK;

    /* 启用 I2S TX/RX 通道 */
    esp_err_t err = i2s_channel_enable(s_es.tx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TX enable 失败: %s", esp_err_to_name(err));
        return err;
    }
    err = i2s_channel_enable(s_es.rx_chan);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "RX enable 失败: %s", esp_err_to_name(err));
        i2s_channel_disable(s_es.tx_chan);
        return err;
    }

    /* PA 控制 */
    if (s_es.pa_pin != GPIO_NUM_NC) {
        gpio_set_level(s_es.pa_pin, 1);
    }

    s_es.running = 1;
    ESP_LOGI(TAG, "ES8311 已启动 (ADC+DAC+PA)");
    return ESP_OK;
}

esp_err_t es8311_stop(void)
{
    if (!s_es.ready) return ESP_ERR_INVALID_STATE;
    if (!s_es.running) return ESP_OK;

    i2s_channel_disable(s_es.rx_chan);
    i2s_channel_disable(s_es.tx_chan);

    if (s_es.pa_pin != GPIO_NUM_NC) {
        gpio_set_level(s_es.pa_pin, 0);
    }

    s_es.running = 0;
    ESP_LOGI(TAG, "ES8311 已停止");
    return ESP_OK;
}

esp_err_t es8311_deinit(void)
{
    if (!s_es.ready) return ESP_OK;
    es8311_stop();

    if (s_es.dev) {
        esp_codec_dev_close(s_es.dev);
        esp_codec_dev_delete(s_es.dev);
        s_es.dev = NULL;
    }
    if (s_es.codec_if) {
        audio_codec_delete_codec_if(s_es.codec_if);
        s_es.codec_if = NULL;
    }
    if (s_es.gpio_if) {
        audio_codec_delete_gpio_if(s_es.gpio_if);
        s_es.gpio_if = NULL;
    }
    if (s_es.ctrl_if) {
        audio_codec_delete_ctrl_if(s_es.ctrl_if);
        s_es.ctrl_if = NULL;
    }
    if (s_es.data_if) {
        audio_codec_delete_data_if(s_es.data_if);
        s_es.data_if = NULL;
    }
    if (s_es.tx_chan) {
        i2s_del_channel(s_es.tx_chan);
        s_es.tx_chan = NULL;
    }
    if (s_es.rx_chan) {
        i2s_del_channel(s_es.rx_chan);
        s_es.rx_chan = NULL;
    }
    if (s_es.i2c_owned && s_es.i2c_bus) {
        i2c_del_master_bus(s_es.i2c_bus);
        s_es.i2c_bus = NULL;
        s_es.i2c_owned = 0;
    }

    s_es.ready = 0;
    ESP_LOGI(TAG, "ES8311 已释放");
    return ESP_OK;
}

esp_err_t es8311_set_volume(int vol)
{
    if (vol < 0) vol = 0;
    if (vol > 100) vol = 100;
    s_es.volume = vol;
    if (s_es.dev) {
        return esp_codec_dev_set_out_vol(s_es.dev, vol);
    }
    return ESP_OK;
}

esp_err_t es8311_set_mic_gain(float gain_db)
{
    if (!s_es.dev) return ESP_ERR_INVALID_STATE;
    return esp_codec_dev_set_in_gain(s_es.dev, gain_db);
}

int es8311_read(int16_t *samples, int max_samples, int timeout_ms)
{
    if (!s_es.ready || !s_es.running) return -1;
    size_t r_bytes = 0;
    size_t req = (size_t)max_samples * sizeof(int16_t);
    esp_err_t err = i2s_channel_read(s_es.rx_chan, samples, req, &r_bytes,
                                     pdMS_TO_TICKS(timeout_ms));
    if (err != ESP_OK) return -1;
    return (int)(r_bytes / sizeof(int16_t));
}

int es8311_write(const int16_t *samples, int num_samples, int timeout_ms)
{
    if (!s_es.ready || !s_es.running) return -1;
    size_t w_bytes = 0;
    size_t req = (size_t)num_samples * sizeof(int16_t);
    esp_err_t err = i2s_channel_write(s_es.tx_chan, samples, req, &w_bytes,
                                      pdMS_TO_TICKS(timeout_ms));
    if (err != ESP_OK) return -1;
    return (int)(w_bytes / sizeof(int16_t));
}

int es8311_is_ready(void)
{
    return s_es.ready;
}
