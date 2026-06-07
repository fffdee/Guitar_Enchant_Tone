/* USB MSC: 把 TF 卡暴露为 U 盘 (tinyusb + diskio_sdmmc).
 * 双核协调: core0 跑 USB 事件, core1 跑推理.
 * 存储互斥: 推理写盘期间调用 usb_msc_set_busy(true), MSC 报介质忙; 完成后 false, PC 自动刷新. */
#ifndef USB_MSC_H
#define USB_MSC_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化 TinyUSB MSC (需在 fs_mount 成功、sd_card_get() 非 NULL 后调用) */
esp_err_t usb_msc_init(void);

/* 通知 USB 介质忙: true = 推理占用 SD 中, PC 访问此时会得到忙响应; false = SD 空闲 */
void usb_msc_set_busy(bool busy);
bool usb_msc_is_busy(void);

/* 作为框架驱动注册 (init = usb_msc_init) */
extern const struct bsp_driver usb_msc_driver;

#ifdef __cplusplus
}
#endif
#endif
