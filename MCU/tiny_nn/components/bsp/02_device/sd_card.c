#include "sd_card.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_ldo_regulator.h"

static const char *TAG = "sd_card";
static sdmmc_card_t            *s_card    = NULL;
static esp_ldo_channel_handle_t s_ldo_sd  = NULL;

/*
 * ESP32-P4: VDD_SD_DPHY (SDMMC IO 线电压) 由片上 LDO 通道 4 提供.
 * 用 esp_ldo_acquire_channel 把通道 4 固定锁在 3.3V (照搬板子官方例程 18_sdcard).
 * 注意: 板子官方例程从不驱动 GPIO45, SD 供电完全靠 LDO; 强驱动 GPIO45 反而
 * 会干扰 4-bit 总线 (表现为 SSR 阶段 timeout), 所以这里不配置 GPIO45.
 */
esp_err_t sd_card_power_on(void)
{
    if (s_ldo_sd == NULL) {
        esp_ldo_channel_config_t ldo_cfg = {
            .chan_id    = SD_PHY_PWR_LDO_CHAN,        /* LDO_VO4 → VDD_SD_DPHY */
            .voltage_mv = SD_PHY_PWR_LDO_VOLTAGE_MV,  /* 3300 mV */
        };
        esp_err_t err = esp_ldo_acquire_channel(&ldo_cfg, &s_ldo_sd);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_ldo_acquire_channel(ch%d) failed: %s",
                     SD_PHY_PWR_LDO_CHAN, esp_err_to_name(err));
            return err;
        }
    }
    vTaskDelay(pdMS_TO_TICKS(10));   /* 等 LDO 稳定 */

    ESP_LOGI(TAG, "SD power on: LDO ch%d=%dmV (GPIO45 not driven, 同官方例程)",
             SD_PHY_PWR_LDO_CHAN, SD_PHY_PWR_LDO_VOLTAGE_MV);
    return ESP_OK;
}

sdmmc_host_t sd_card_host(void)
{
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.slot         = SD_SLOT;
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;  /* 40MHz, 同官方例程 */
    return host;
}

sdmmc_slot_config_t sd_card_slot(void)
{
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = SD_BUS_WIDTH;
    slot.clk   = SD_PIN_CLK;
    slot.cmd   = SD_PIN_CMD;
    slot.d0    = SD_PIN_D0;
    slot.d1    = SD_PIN_D1;
    slot.d2    = SD_PIN_D2;
    slot.d3    = SD_PIN_D3;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    return slot;
}

void sd_card_set(sdmmc_card_t *card) { s_card = card; }
sdmmc_card_t *sd_card_get(void) { return s_card; }

const bsp_driver_t sd_card_driver = {
    .name   = "sdcard",
    .init   = sd_card_power_on,
    .deinit = NULL,
};
