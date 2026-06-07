/* USB MSC: 把 TF 卡暴露为 U 盘 (tinyusb + diskio_sdmmc).
 * 依赖托管组件 espressif/esp_tinyusb (见同目录 README / idf_component.yml).
 * 默认未接入构建; 启用见 bsp/CMakeLists.txt 注释. */
#include "usb_msc.h"
#include "sd_card.h"
#include "drv_device.h"
#include <string.h>
#include <stdatomic.h>
#include "esp_log.h"
#include "esp_check.h"
#include "sdmmc_cmd.h"
#include "tinyusb.h"
#include "tinyusb_msc.h"

static const char *TAG = "usb_msc";
static atomic_bool s_busy = false;

void usb_msc_set_busy(bool busy) { atomic_store(&s_busy, busy); }
bool usb_msc_is_busy(void)       { return atomic_load(&s_busy); }

/* 推理写盘期间报「介质未就绪」, PC 等推理结束后自动重试刷新 (存储自动协调) */
bool tud_msc_is_writable_cb(uint8_t lun) { (void)lun; return !usb_msc_is_busy(); }

void tud_msc_inquiry_cb(uint8_t lun, uint8_t vendor_id[8], uint8_t product_id[16], uint8_t product_rev[4])
{
    (void)lun;
    memcpy(vendor_id, "ESP32   ", 8);
    memcpy(product_id, "XFORM SD Card   ", 16);
    memcpy(product_rev, "1.00", 4);
}

bool tud_msc_test_unit_ready_cb(uint8_t lun)
{
    (void)lun;
    return sd_card_get() != NULL && !usb_msc_is_busy();
}

void tud_msc_capacity_cb(uint8_t lun, uint32_t *block_count, uint16_t *block_size)
{
    (void)lun;
    sdmmc_card_t *card = sd_card_get();
    if (card) { *block_count = card->csd.capacity; *block_size = card->csd.sector_size; }
    else { *block_count = 0; *block_size = 512; }
}

int32_t tud_msc_read10_cb(uint8_t lun, uint32_t lba, uint32_t offset, void *buf, uint32_t bufsize)
{
    (void)lun; (void)offset;
    sdmmc_card_t *card = sd_card_get();
    if (!card || usb_msc_is_busy()) return -1;
    esp_err_t err = sdmmc_read_sectors(card, buf, lba, bufsize / card->csd.sector_size);
    return (err == ESP_OK) ? (int32_t)bufsize : -1;
}

int32_t tud_msc_write10_cb(uint8_t lun, uint32_t lba, uint32_t offset, uint8_t *buf, uint32_t bufsize)
{
    (void)lun; (void)offset;
    sdmmc_card_t *card = sd_card_get();
    if (!card || usb_msc_is_busy()) return -1;
    esp_err_t err = sdmmc_write_sectors(card, buf, lba, bufsize / card->csd.sector_size);
    return (err == ESP_OK) ? (int32_t)bufsize : -1;
}

int32_t tud_msc_scsi_cb(uint8_t lun, uint8_t const scsi_cmd[16], void *buf, uint16_t bufsize)
{
    (void)lun; (void)buf; (void)bufsize; (void)scsi_cmd;
    return -1;
}

esp_err_t usb_msc_init(void)
{
    sdmmc_card_t *card = sd_card_get();
    if (!card) { ESP_LOGE(TAG, "SD not mounted, skip USB MSC"); return ESP_ERR_INVALID_STATE; }

    const tinyusb_config_t tusb_cfg = { 0 };   /* 用默认描述符 */
    ESP_RETURN_ON_ERROR(tinyusb_driver_install(&tusb_cfg), TAG, "tusb install");
    ESP_LOGI(TAG, "USB MSC ready (TF 卡已暴露为 U 盘)");
    return ESP_OK;
}

const bsp_driver_t usb_msc_driver = {
    .name = "usb_msc",
    .init = usb_msc_init,
    .deinit = NULL,
};
