/* bsp_init: 驱动注册 -> 按序初始化 -> 文件系统挂载 (仿 banux drv_init) */
#include "bsp.h"
#include "drv_device.h"
#include "sd_card.h"
#include "fs_mount.h"
#include "esp_log.h"
#include "esp_check.h"

static const char *TAG = "bsp";

esp_err_t bsp_init(void)
{
    ESP_LOGI(TAG, "BSP init...");
    /* 1) 注册底层设备驱动 (可继续扩展: USB/LCD/...) */
    ESP_RETURN_ON_ERROR(drv_register(&sd_card_driver), TAG, "register sdcard");
    /* 2) 按注册顺序初始化 (SD 上电等) */
    ESP_RETURN_ON_ERROR(drv_init_all(), TAG, "drv_init_all");
    /* 3) 挂载文件系统 */
    ESP_RETURN_ON_ERROR(fs_mount(), TAG, "fs_mount");
    ESP_LOGI(TAG, "BSP ready (%d drivers, %s mounted)", drv_count(), BSP_SD_MOUNT);
    return ESP_OK;
}
