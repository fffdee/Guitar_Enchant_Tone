/*
 * ESP-GT-XFORM @ ESP32-P4 主程序。
 * 开机流程: BSP 初始化(SD+FATFS) -> 加载模型包 -> 启动推理 worker(core1) + 命令行(core0)。
 * 推理由命令行触发, 推理进行时命令行依旧可用 (双核解耦)。
 *
 * 命令行示例:
 *   model                      查看模型信息(乐器列表/采样率/参数量)
 *   infer bass -p -12 -g 9 -c soft   转换为 bass: 降12半音 +9dB 软限幅
 *   infer banjo                转换为 banjo (模型若无该乐器则跳过)
 *   status / ls /sdcard/out    查看进度 / 取结果
 *
 * TF 卡放置:
 *   /sdcard/model/xform_model.bin   (PC 端 export_model_package 产物)
 *   /sdcard/in/guitar.wav           (输入吉他, 48k 单声道)
 *   /sdcard/out/                    (输出目录, 推理时自动创建)
 */
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "bsp.h"
#include "audio_xform.h"
#include "infer_worker.h"
#include "cli.h"

static const char *TAG = "app";

#define MODEL_PATH  BSP_SD_MOUNT "/model/xform_model.bin"

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

    if (bsp_init() != ESP_OK) {
        ESP_LOGE(TAG, "BSP 初始化失败");
        return;
    }

    /* 加载模型 (失败不致命: 命令行仍启动, 便于排查/换卡后重试) */
    if (audio_xform_init(MODEL_PATH) != ESP_OK)
        ESP_LOGW(TAG, "模型加载失败 (检查 %s), 可放好文件后用命令行重试", MODEL_PATH);
    else
        audio_xform_print_model();

    /* core1 推理 worker + core0 命令行 */
    ESP_ERROR_CHECK(infer_worker_start());
    ESP_ERROR_CHECK(cli_start());

    ESP_LOGI(TAG, "就绪: 输入 'help' 查看命令, 'infer bass -p -12 -g 9 -c soft' 开始转换");
}
