/* SD 卡设备驱动 (SDMMC 4-bit, slot1). 引脚: D0-3=GPIO39-42, CLK=43, CMD=44, PWR_EN=45 */
#ifndef SD_CARD_H
#define SD_CARD_H

#include "esp_err.h"
#include "driver/sdmmc_host.h"
#include "drv_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 引脚定义 (按硬件) */
#define SD_PIN_D0      39
#define SD_PIN_D1      40
#define SD_PIN_D2      41
#define SD_PIN_D3      42
#define SD_PIN_CLK     43
#define SD_PIN_CMD     44
#define SD_PIN_PWR_EN  45
#define SD_SLOT        SDMMC_HOST_SLOT_1
#define SD_BUS_WIDTH   4

/* VDD_SD_DPHY 供电: 片上 LDO 通道 4 固定 3.3V (同板子官方例程 18_sdcard) */
#define SD_PHY_PWR_LDO_CHAN        4
#define SD_PHY_PWR_LDO_VOLTAGE_MV  3300

/* 作为框架驱动注册 (init = 上电 + GPIO 配置) */
extern const bsp_driver_t sd_card_driver;

/* PWR_EN 上电 (高电平使能) */
esp_err_t sd_card_power_on(void);

/* 取已配置的 host / slot (供 fs_mount 使用) */
sdmmc_host_t        sd_card_host(void);
sdmmc_slot_config_t sd_card_slot(void);

/* 挂载后由 fs_mount 写入 card 句柄; USB MSC 等可取用 */
void  sd_card_set(sdmmc_card_t *card);
sdmmc_card_t *sd_card_get(void);

#ifdef __cplusplus
}
#endif
#endif
