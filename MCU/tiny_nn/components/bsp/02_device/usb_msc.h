/* USB MSC: 把 TF 卡暴露为 U 盘 (esp_tinyusb + SDMMC storage).
 * 双核协调: core0 跑 USB 事件, core1 跑推理.
 * 存储互斥: APP 与 USB Host 通过 mount point 切换独占 SD 卡 FAT 卷. */
#ifndef USB_MSC_H
#define USB_MSC_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化 SDMMC + TinyUSB MSC, 默认挂载到 APP (/sdcard) */
esp_err_t usb_msc_init(void);

typedef enum {
    USB_MSC_MOUNT_UNKNOWN = 0,
    USB_MSC_MOUNT_APP,
    USB_MSC_MOUNT_USB,
} usb_msc_mount_t;

/* 通知 USB 介质忙: true = 推理占用 SD 中, PC 访问此时会得到忙响应; false = SD 空闲 */
void usb_msc_set_busy(bool busy);
bool usb_msc_is_busy(void);
bool usb_msc_is_ready(void);
bool usb_msc_is_attached(void);
bool usb_msc_console_ready(void);
int usb_msc_port(void);

/* APP 模式: MCU 可访问 /sdcard; USB 模式: PC 可作为 U 盘访问, MCU 侧暂不可读写 */
esp_err_t usb_msc_mount_app(void);
esp_err_t usb_msc_mount_usb(void);
usb_msc_mount_t usb_msc_mount_point(void);
const char *usb_msc_mount_name(void);

/* 作为框架驱动注册 (init = usb_msc_init) */
extern const struct bsp_driver usb_msc_driver;

#ifdef __cplusplus
}
#endif
#endif
