/*
 * BSP 公共入口 (仿 banux 分层框架, ESP-IDF 原生实现)。
 * 层次: 02_device(SD) -> 03_framework(注册/init) -> 05_component(FATFS/WAV/模型) -> 06_app(音色转换)
 * 与 Tinynn 推理框架解耦: BSP 只通过 model_store/audio_xform 调用 Tinynn 公共 API。
 */
#ifndef BSP_H
#define BSP_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 一键初始化: 驱动框架 -> SD 卡上电+初始化 -> FATFS 挂载 /sdcard */
esp_err_t bsp_init(void);

/* SD 挂载点 */
#define BSP_SD_MOUNT "/sdcard"

#ifdef __cplusplus
}
#endif
#endif
