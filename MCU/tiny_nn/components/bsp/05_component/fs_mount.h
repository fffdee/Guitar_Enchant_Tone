/* FATFS 挂载 (基于 SD 卡) */
#ifndef FS_MOUNT_H
#define FS_MOUNT_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t fs_mount(void);     /* esp_vfs_fat_sdmmc_mount 到 BSP_SD_MOUNT */
void      fs_unmount(void);
bool      fs_is_mounted(void);

#ifdef __cplusplus
}
#endif
#endif
