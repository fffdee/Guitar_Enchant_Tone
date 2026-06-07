/* 驱动框架: 设备注册与按序初始化 (仿 banux 03_driver_framework, 精简到够用) */
#ifndef DRV_DEVICE_H
#define DRV_DEVICE_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bsp_driver {
    const char *name;
    esp_err_t (*init)(void);     /* 初始化, 返回 ESP_OK 成功 */
    void      (*deinit)(void);   /* 可为 NULL */
} bsp_driver_t;

/* 注册驱动 (按注册顺序初始化). 返回 ESP_OK 成功. */
esp_err_t drv_register(const bsp_driver_t *drv);

/* 依次初始化所有已注册驱动. 任一失败返回其错误码. */
esp_err_t drv_init_all(void);

const bsp_driver_t *drv_find(const char *name);
int drv_count(void);

#ifdef __cplusplus
}
#endif
#endif
