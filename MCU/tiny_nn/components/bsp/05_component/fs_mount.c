#include "fs_mount.h"
#include "bsp.h"
#include "sd_card.h"
#include "esp_vfs_fat.h"
#include "esp_log.h"
#include <inttypes.h>

static const char *TAG = "fs_mount";
static bool s_mounted = false;

esp_err_t fs_mount(void)
{
    if (s_mounted) return ESP_OK;

    esp_vfs_fat_sdmmc_mount_config_t mcfg = {
        .format_if_mount_failed = false,
        .max_files = 6,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_host_t host = sd_card_host();
    sdmmc_slot_config_t slot = sd_card_slot();
    sdmmc_card_t *card = NULL;

    /* 最多重试 3 次 (timeout 0x107 时卡通常第 2 次就成功) */
    esp_err_t err = ESP_FAIL;
    for (int attempt = 1; attempt <= 3; ++attempt) {
        err = esp_vfs_fat_sdmmc_mount(BSP_SD_MOUNT, &host, &slot, &mcfg, &card);
        if (err == ESP_OK) break;
        ESP_LOGW(TAG, "mount attempt %d failed: %s", attempt, esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(200 * attempt));
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mount %s failed after retries: %s", BSP_SD_MOUNT, esp_err_to_name(err));
        return err;
    }
    sd_card_set(card);
    s_mounted = true;
    ESP_LOGI(TAG, "mounted %s (%" PRIu64 "MB)", BSP_SD_MOUNT,
             ((uint64_t)card->csd.capacity * card->csd.sector_size) / (1024ULL * 1024ULL));
    return ESP_OK;
}

void fs_unmount(void)
{
    if (!s_mounted) return;
    esp_vfs_fat_sdcard_unmount(BSP_SD_MOUNT, sd_card_get());
    sd_card_set(NULL);
    s_mounted = false;
    ESP_LOGI(TAG, "unmounted %s", BSP_SD_MOUNT);
}

bool fs_is_mounted(void) { return s_mounted; }
