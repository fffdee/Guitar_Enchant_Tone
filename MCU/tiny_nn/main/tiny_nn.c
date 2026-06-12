/*
 * ESP-GT-XFORM @ ESP32-P4 主程序。
 *
 * 两种运行模式:
 *   1. SD 卡模式 (默认): 从 SD 卡读取 WAV -> 推理 -> 写回 SD 卡
 *   2. I2S 实时模式: I2S RX 采集音频 -> 推理 -> I2S TX 输出
 *
 * 启动时通过 GPIO0 (BOOT 按钮) 选择模式:
 *   - 按住 BOOT 启动: I2S 实时模式
 *   - 正常启动: SD 卡模式
 * 也可在命令行中随时切换:
 *   mode sd    切换到 SD 卡模式
 *   mode i2s   切换到 I2S 实时模式
 *
 * 命令行示例:
 *   model                      查看模型信息(乐器列表/采样率/参数量)
 *   infer bass -p -12 -g 9 -c soft   SD卡模式: 转换为 bass
 *   mode i2s                   切换到 I2S 实时模式
 *   live bass -p -12 -g 9      I2S模式: 实时转换为 bass
 *   live stop                  停止实时推理
 *   status / ls /sdcard/out    查看进度 / 取结果
 *
 * TF 卡放置:
 *   /sdcard/model/xform_model.bin   (PC 端 export_model_package 产物)
 *   /sdcard/in/guitar.wav           (输入吉他, 48k 单声道)
 *   /sdcard/out/                    (输出目录, 推理时自动创建)
 */
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "bsp.h"
#include "audio_xform.h"
#include "infer_worker.h"
#include "i2s_driver.h"
#include "i2s_xform.h"
#include "cli.h"

static const char *TAG = "app";

#define MODEL_PATH  BSP_SD_MOUNT "/model/xform_model.bin"
#define BOOT_BTN    GPIO_NUM_0   /* BOOT 按钮, 用于选择启动模式 */

/* 运行模式 */
typedef enum {
    RUN_MODE_SD,       /* SD 卡文件模式 */
    RUN_MODE_I2S,      /* I2S 实时模式 */
} run_mode_t;

static run_mode_t s_mode = RUN_MODE_SD;

/* 检测 BOOT 按钮是否按下 (低电平有效) */
static int boot_btn_pressed(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BOOT_BTN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    return (gpio_get_level(BOOT_BTN) == 0);
}

void app_main(void)
{
    ESP_LOGI(TAG, "ESP-GT-XFORM (ESP32-P4) 启动");

    /* PSRAM 诊断: 模型 ~583KB + 网络权重需从 PSRAM 分配 */
    size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "PSRAM total=%uKB free=%uKB | 内部RAM free=%uKB",
             (unsigned)(psram_total / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    if (psram_total == 0)
        ESP_LOGE(TAG, "PSRAM 未初始化! 检查 sdkconfig SPIRAM");

    /* 检测启动模式: 按住 BOOT 按钮启动 -> I2S 实时模式 */
    if (boot_btn_pressed()) {
        s_mode = RUN_MODE_I2S;
        ESP_LOGI(TAG, "检测到 BOOT 按键: I2S 实时模式");
    } else {
        s_mode = RUN_MODE_SD;
        ESP_LOGI(TAG, "SD 卡文件模式 (按住 BOOT 启动可切换 I2S 实时模式)");
    }

    /* BSP 初始化 (SD 卡 + FATFS) */
    if (bsp_init() != ESP_OK) {
        ESP_LOGE(TAG, "BSP 初始化失败");
        /* I2S 模式不依赖 SD 卡, 可以继续 */
        if (s_mode == RUN_MODE_SD) return;
    }

    /* 加载模型 (两种模式都需要) */
    if (audio_xform_init(MODEL_PATH) != ESP_OK)
        ESP_LOGW(TAG, "模型加载失败 (检查 %s), 可放好文件后用命令行重试", MODEL_PATH);
    else
        audio_xform_print_model();

    /* I2S 模式: 初始化驱动并自动启动实时推理 */
    if (s_mode == RUN_MODE_I2S && audio_xform_loaded()) {
        if (i2s_driver_init(0) == ESP_OK) {
            i2s_xform_cfg_t cfg = {
                .instrument = "bass",
                .pitch_semitones = 0.0f,
                .gain_db = 0.0f,
                .clip_mode = AUDIO_CLIP_LIMIT,
                .add_noise = 0,
            };
            if (i2s_xform_start(&cfg) == ESP_OK) {
                ESP_LOGI(TAG, "I2S 实时推理已启动 (默认乐器: bass)");
            } else {
                ESP_LOGE(TAG, "I2S 实时推理启动失败");
            }
        } else {
            ESP_LOGE(TAG, "I2S 初始化失败");
        }
    }

    /* SD 卡模式: 启动推理 worker */
    if (s_mode == RUN_MODE_SD) {
        ESP_ERROR_CHECK(infer_worker_start());
    }

    /* 命令行始终启动 (两种模式都需要) */
    ESP_ERROR_CHECK(cli_start());

    ESP_LOGI(TAG, "就绪: 输入 'help' 查看命令");
    if (s_mode == RUN_MODE_SD)
        ESP_LOGI(TAG, "  'infer bass -p -12 -g 9 -c soft' 开始转换");
    else
        ESP_LOGI(TAG, "  'live stop' 停止实时推理, 'mode sd' 切换模式");
}
