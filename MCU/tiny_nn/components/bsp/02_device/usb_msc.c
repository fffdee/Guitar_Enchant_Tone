/* USB MSC: 把 TF 卡暴露为 U 盘 (esp_tinyusb + SDMMC storage).
 * 说明:
 * - 不手写 tud_msc_* 底层回调, 避免与 esp_tinyusb/tinyusb_msc.c 重复定义.
 * - SD 卡所有权由 tinyusb_msc_set_storage_mount_point() 在 APP 与 USB Host 间切换.
 */
#include "usb_msc.h"
#include "bsp.h"
#include "sd_card.h"
#include "fs_mount.h"
#include "drv_device.h"
#include <inttypes.h>
#include <stdlib.h>
#include <stdatomic.h>
#include "esp_log.h"
#include "esp_check.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "tinyusb.h"
#include "tinyusb_default_config.h"
#include "tinyusb_cdc_acm.h"
#include "tinyusb_console.h"
#include "tinyusb_msc.h"

static const char *TAG = "usb_msc";
static atomic_bool s_busy = false;   /* APP 正在推理/写盘, 禁止切到 USB Host */
static atomic_bool s_ready = false;
static atomic_bool s_attached = false;
static atomic_bool s_console_ready = false;
static tinyusb_msc_storage_handle_t s_storage = NULL;
static usb_msc_mount_t s_mount = USB_MSC_MOUNT_UNKNOWN;
static int s_port = -1;

void usb_msc_set_busy(bool busy) { atomic_store(&s_busy, busy); }
bool usb_msc_is_busy(void)       { return atomic_load(&s_busy); }
bool usb_msc_is_ready(void)      { return atomic_load(&s_ready); }
bool usb_msc_is_attached(void)   { return atomic_load(&s_attached); }
bool usb_msc_console_ready(void) { return atomic_load(&s_console_ready); }
int usb_msc_port(void)           { return s_port; }
usb_msc_mount_t usb_msc_mount_point(void) { return s_mount; }

const char *usb_msc_mount_name(void)
{
    switch (s_mount) {
    case USB_MSC_MOUNT_APP: return "app";
    case USB_MSC_MOUNT_USB: return "usb";
    default: return "unknown";
    }
}

static esp_err_t ensure_sd_card_ready(void)
{
    if (sd_card_get()) return ESP_OK;

    sdmmc_host_t host = sd_card_host();
    sdmmc_slot_config_t slot = sd_card_slot();

    ESP_RETURN_ON_ERROR(sdmmc_host_init(), TAG, "sdmmc_host_init");
    ESP_RETURN_ON_ERROR(sdmmc_host_init_slot(host.slot, &slot), TAG, "sdmmc_host_init_slot");

    sdmmc_card_t *card = (sdmmc_card_t *)calloc(1, sizeof(sdmmc_card_t));
    ESP_RETURN_ON_FALSE(card != NULL, ESP_ERR_NO_MEM, TAG, "alloc sdmmc_card_t");

    esp_err_t err = sdmmc_card_init(&host, card);
    if (err != ESP_OK) {
        free(card);
        ESP_LOGE(TAG, "sdmmc_card_init failed: %s", esp_err_to_name(err));
        return err;
    }

    sd_card_set(card);
    ESP_LOGI(TAG, "SDMMC card initialized for USB MSC (%" PRIu64 " MB)",
             ((uint64_t)card->csd.capacity * card->csd.sector_size) / (1024ULL * 1024ULL));
    return ESP_OK;
}

static void storage_event_cb(tinyusb_msc_storage_handle_t handle, tinyusb_msc_event_t *event, void *arg)
{
    (void)handle;
    (void)arg;
    if (!event) return;

    if (event->id == TINYUSB_MSC_EVENT_MOUNT_COMPLETE) {
        s_mount = (event->mount_point == TINYUSB_MSC_STORAGE_MOUNT_USB)
                      ? USB_MSC_MOUNT_USB
                      : USB_MSC_MOUNT_APP;
        ESP_LOGI(TAG, "MSC storage mounted to %s", usb_msc_mount_name());
    } else if (event->id == TINYUSB_MSC_EVENT_MOUNT_FAILED ||
               event->id == TINYUSB_MSC_EVENT_FORMAT_FAILED ||
               event->id == TINYUSB_MSC_EVENT_FORMAT_REQUIRED) {
        ESP_LOGW(TAG, "MSC storage event=%d mount=%d", (int)event->id, (int)event->mount_point);
    }
}

static void device_event_cb(tinyusb_event_t *event, void *arg)
{
    (void)arg;
    if (!event) return;
    if (event->id == TINYUSB_EVENT_ATTACHED) {
        atomic_store(&s_attached, true);
    } else if (event->id == TINYUSB_EVENT_DETACHED) {
        atomic_store(&s_attached, false);
    }
    ESP_LOGI(TAG, "TinyUSB event=%d port=%d", (int)event->id, (int)event->rhport);
}

esp_err_t usb_msc_mount_app(void)
{
    if (!s_storage) return ESP_ERR_INVALID_STATE;
    esp_err_t err = tinyusb_msc_set_storage_mount_point(s_storage, TINYUSB_MSC_STORAGE_MOUNT_APP);
    if (err == ESP_OK) s_mount = USB_MSC_MOUNT_APP;
    return err;
}

esp_err_t usb_msc_mount_usb(void)
{
    if (!s_storage) return ESP_ERR_INVALID_STATE;
    if (usb_msc_is_busy()) return ESP_ERR_INVALID_STATE;
    esp_err_t err = tinyusb_msc_set_storage_mount_point(s_storage, TINYUSB_MSC_STORAGE_MOUNT_USB);
    if (err == ESP_OK) s_mount = USB_MSC_MOUNT_USB;
    return err;
}

esp_err_t usb_msc_init(void)
{
    if (usb_msc_is_ready()) return ESP_OK;

    /* 普通 FATFS 挂载与 USB MSC storage 不能同时拥有同一张卡。 */
    if (fs_is_mounted()) {
        fs_unmount();
    }

    ESP_RETURN_ON_ERROR(ensure_sd_card_ready(), TAG, "init sd card");
    sdmmc_card_t *card = sd_card_get();
    ESP_RETURN_ON_FALSE(card != NULL, ESP_ERR_INVALID_STATE, TAG, "sd card is not ready");

    tinyusb_msc_driver_config_t driver_cfg = {
        .callback = storage_event_cb,
        .callback_arg = NULL,
    };
    driver_cfg.user_flags.auto_mount_off = 1;  /* 避免 USB 插入时自动抢走 /sdcard */
    ESP_RETURN_ON_ERROR(tinyusb_msc_install_driver(&driver_cfg), TAG, "msc driver install");

    tinyusb_msc_storage_config_t storage_cfg = {
        .medium.card = card,
        .mount_point = TINYUSB_MSC_STORAGE_MOUNT_APP,
        .fat_fs = {
            .base_path = BSP_SD_MOUNT,
            .config = {
                .format_if_mount_failed = false,
                .max_files = 6,
                .allocation_unit_size = 16 * 1024,
            },
            .do_not_format = true,
            .format_flags = 0,
        },
    };
    ESP_RETURN_ON_ERROR(tinyusb_msc_new_storage_sdmmc(&storage_cfg, &s_storage), TAG, "msc sdmmc storage");
    s_mount = USB_MSC_MOUNT_APP;

    tinyusb_config_t tusb_cfg = TINYUSB_DEFAULT_CONFIG(device_event_cb, NULL);
    tusb_cfg.task.xCoreID = 0;  /* core1 留给推理任务 */
    ESP_RETURN_ON_ERROR(tinyusb_driver_install(&tusb_cfg), TAG, "tinyusb install");
    s_port = (int)tusb_cfg.port;

    tinyusb_config_cdcacm_t cdc_cfg = {
        .cdc_port = TINYUSB_CDC_ACM_0,
        .callback_rx = NULL,
        .callback_rx_wanted_char = NULL,
        .callback_line_state_changed = NULL,
        .callback_line_coding_changed = NULL,
    };
    ESP_RETURN_ON_ERROR(tinyusb_cdcacm_init(&cdc_cfg), TAG, "cdc acm init");
    /* Keep esp_console on the primary UART/USB-Serial-JTAG monitor.
     * Redirecting stdio to TinyUSB CDC hides the CLI when the OTG port is not
     * connected or not enumerated yet, making "usb host" impossible to type. */
    atomic_store(&s_console_ready, true);

    atomic_store(&s_ready, true);
    ESP_LOGI(TAG, "TinyUSB CDC+MSC ready: port=%d mount=%s path=%s, CLI stays on primary console; use 'usb host' to expose SD",
             s_port, usb_msc_mount_name(), BSP_SD_MOUNT);
    return ESP_OK;
}

const bsp_driver_t usb_msc_driver = {
    .name = "usb_msc",
    .init = usb_msc_init,
    .deinit = NULL,
};
