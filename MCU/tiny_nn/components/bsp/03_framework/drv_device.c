#include "drv_device.h"
#include <string.h>
#include "esp_log.h"

#define DRV_MAX 16
static const char *TAG = "drv";
static const bsp_driver_t *s_drivers[DRV_MAX];
static int s_count = 0;

esp_err_t drv_register(const bsp_driver_t *drv)
{
    if (!drv || !drv->name) return ESP_ERR_INVALID_ARG;
    if (s_count >= DRV_MAX) return ESP_ERR_NO_MEM;
    for (int i = 0; i < s_count; ++i)
        if (s_drivers[i] == drv) return ESP_OK;   /* 去重 */
    s_drivers[s_count++] = drv;
    return ESP_OK;
}

esp_err_t drv_init_all(void)
{
    for (int i = 0; i < s_count; ++i) {
        const bsp_driver_t *d = s_drivers[i];
        if (!d->init) continue;
        esp_err_t err = d->init();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "driver '%s' init failed: %s", d->name, esp_err_to_name(err));
            return err;
        }
        ESP_LOGI(TAG, "driver '%s' init ok", d->name);
    }
    return ESP_OK;
}

const bsp_driver_t *drv_find(const char *name)
{
    if (!name) return NULL;
    for (int i = 0; i < s_count; ++i)
        if (strcmp(s_drivers[i]->name, name) == 0) return s_drivers[i];
    return NULL;
}

int drv_count(void) { return s_count; }
