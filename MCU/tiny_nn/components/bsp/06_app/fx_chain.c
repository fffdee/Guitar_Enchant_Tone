/* 电吉他效果器链实时处理实现。
 *
 * 数据流:
 *   I2S RX (16bit) -> int16 转 float -> 输入增益
 *   -> 依次通过效果器链 (in-place) -> 输出增益 -> float 转 int16 -> I2S TX
 *
 * 帧长: 默认 240 采样 @48kHz = 5ms/帧, 与 i2s_xform 一致。
 * 延迟: ~2 帧 (输入缓冲 + 处理) ≈ 10ms。
 *
 * 线程安全:
 *   链操作 (add/del/clear/set) 通过 s_chain.lock 互斥锁保护。
 *   实时任务在处理每帧前取锁拷贝链指针, 处理期间不持锁。
 *   修改链时短暂持锁, 不会阻塞实时任务超过一帧。
 *
 * 与 i2s_xform 互斥: 两者共用 I2S 驱动, 不能同时运行。
 *   fx_chain_start() 会检查 i2s_xform_running(), 反之亦然 (在 cli.c 中保证)。
 */
#include "fx_chain.h"
#include "fx_modules.h"
#include "i2s_driver.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <inttypes.h>

static const char *TAG = "fx_chain";

#define FX_TASK_STACK   8192
#define FX_TASK_PRIO    7       /* 与 i2s_xform 同优先级, 互斥运行 */
#define FX_TASK_CORE    1       /* core1 实时处理 */

/* 运行时状态 */
static struct {
    TaskHandle_t      task;
    SemaphoreHandle_t lock;        /* 保护 chain[] 与 count */
    fx_chain_cfg_t    cfg;
    fx_node_t        *chain[FX_CHAIN_MAX];
    int               count;
    volatile int      running;     /* 独立模式运行标志 */
    volatile int      stop_req;
    volatile int      global_bypass; /* 全局旁通 (两种模式共用) */
    /* 统计 */
    uint32_t          frames;
    int64_t           sum_us;
    int64_t           min_us;
    int64_t           max_us;
    int               underruns;
    int               overruns;
} s_fx;

/* int16 <-> float 转换 */
static inline float i16_to_f(int16_t s) { return (float)s / 32768.0f; }
static inline int16_t f_to_i16(float f) {
    if (f >= 1.0f)  return 32767;
    if (f <= -1.0f) return -32768;
    return (int16_t)(int32_t)(f * 32768.0f);
}

/* 确保互斥锁已创建 (懒初始化, 串联模式也需要) */
static void ensure_lock(void)
{
    if (!s_fx.lock) {
        s_fx.lock = xSemaphoreCreateMutex();
    }
}

/* 实时处理任务 */
static void rt_task(void *arg)
{
    (void)arg;
    int hop = s_fx.cfg.hop_size;
    int sr  = s_fx.cfg.sample_rate;
    float in_gain  = powf(10.0f, s_fx.cfg.input_gain_db / 20.0f);
    float out_gain = powf(10.0f, s_fx.cfg.output_gain_db / 20.0f);

    /* 分配缓冲 (i2s_buf 内部 RAM, in_buf PSRAM) */
    int16_t *i2s_buf = (int16_t *)heap_caps_malloc((size_t)hop * sizeof(int16_t),
                                                    MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    float   *buf     = (float *)heap_caps_malloc((size_t)hop * sizeof(float),
                                                  MALLOC_CAP_SPIRAM);
    if (!i2s_buf || !buf) {
        ESP_LOGE(TAG, "缓冲分配失败");
        goto cleanup;
    }

    /* 启动 I2S 收发 */
    esp_err_t err = i2s_driver_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S 启动失败: %s", esp_err_to_name(err));
        goto cleanup;
    }

    ESP_LOGI(TAG, "效果器链启动: hop=%d sr=%d 链长=%d", hop, sr, s_fx.count);
    s_fx.running = 1;

    /* 主循环: 读一帧 -> 效果器链处理 -> 写一帧 */
    while (!s_fx.stop_req) {
        /* 1. 从 I2S RX 读取一帧 */
        int got = i2s_driver_read(i2s_buf, hop, 100);
        if (got < hop) {
            s_fx.underruns++;
            for (int i = (got > 0 ? got : 0); i < hop; i++)
                i2s_buf[i] = 0;
        }

        /* 2. int16 -> float + 输入增益 */
        for (int i = 0; i < hop; i++)
            buf[i] = i16_to_f(i2s_buf[i]) * in_gain;

        /* 3. 效果器链处理 (取锁拷贝指针, 处理期间不持锁) */
        fx_node_t *chain_snap[FX_CHAIN_MAX];
        int count_snap;
        xSemaphoreTake(s_fx.lock, portMAX_DELAY);
        count_snap = s_fx.count;
        memcpy(chain_snap, s_fx.chain, sizeof(chain_snap));
        xSemaphoreGive(s_fx.lock);

        int64_t t0 = esp_timer_get_time();
        for (int k = 0; k < count_snap; k++) {
            fx_node_t *node = chain_snap[k];
            if (node && node->enabled && node->process) {
                node->process(node, buf, hop);
            }
        }
        int64_t dt_us = esp_timer_get_time() - t0;

        /* 4. 输出增益 + float -> int16 */
        for (int i = 0; i < hop; i++)
            i2s_buf[i] = f_to_i16(buf[i] * out_gain);

        /* 5. 写入 I2S TX */
        int written = i2s_driver_write(i2s_buf, hop, 10);
        if (written < hop)
            s_fx.overruns++;

        /* 6. 更新统计 */
        s_fx.frames++;
        s_fx.sum_us += dt_us;
        if (dt_us < s_fx.min_us) s_fx.min_us = dt_us;
        if (dt_us > s_fx.max_us) s_fx.max_us = dt_us;
    }

    /* 停止 I2S */
    i2s_driver_stop();
    s_fx.running = 0;
    ESP_LOGI(TAG, "效果器链已停止: %" PRIu32 "帧 avg=%" PRId64 "us under=%d over=%d",
             s_fx.frames, s_fx.frames > 0 ? s_fx.sum_us / s_fx.frames : 0,
             s_fx.underruns, s_fx.overruns);

cleanup:
    if (i2s_buf) heap_caps_free(i2s_buf);
    if (buf)     heap_caps_free(buf);
    s_fx.running = 0;
    s_fx.task = NULL;
    vTaskDelete(NULL);
}

/* ============================================================
 * 启停
 * ============================================================ */
esp_err_t fx_chain_start(const fx_chain_cfg_t *cfg)
{
    if (!cfg) return ESP_ERR_INVALID_ARG;
    if (s_fx.running) {
        ESP_LOGW(TAG, "已在运行, 请先停止");
        return ESP_ERR_INVALID_STATE;
    }
    if (!i2s_driver_is_ready()) {
        ESP_LOGE(TAG, "I2S 未初始化");
        return ESP_ERR_INVALID_STATE;
    }

    /* 重置状态 (保留已有效果器链) */
    s_fx.stop_req = 0;
    s_fx.frames = 0;
    s_fx.sum_us = 0;
    s_fx.min_us = INT64_MAX;
    s_fx.max_us = 0;
    s_fx.underruns = 0;
    s_fx.overruns = 0;
    memcpy(&s_fx.cfg, cfg, sizeof(s_fx.cfg));
    if (s_fx.cfg.hop_size <= 0) s_fx.cfg.hop_size = FX_HOP_DEFAULT;
    if (s_fx.cfg.sample_rate <= 0) s_fx.cfg.sample_rate = I2S_DEFAULT_SAMPLE_RATE;

    if (!s_fx.lock) {
        ensure_lock();
        if (!s_fx.lock) return ESP_ERR_NO_MEM;
    }

    BaseType_t ok = xTaskCreatePinnedToCore(rt_task, "fx_chain", FX_TASK_STACK,
                                            NULL, FX_TASK_PRIO, &s_fx.task, FX_TASK_CORE);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "任务创建失败");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t fx_chain_stop(void)
{
    if (!s_fx.running && !s_fx.task) return ESP_OK;
    s_fx.stop_req = 1;
    for (int i = 0; i < 200 && s_fx.running; i++)
        vTaskDelay(pdMS_TO_TICKS(10));
    return ESP_OK;
}

int fx_chain_running(void)
{
    return s_fx.running;
}

/* ============================================================
 * 串联模式: 缓冲处理 (供 i2s_xform 调用)
 * ============================================================ */
esp_err_t fx_chain_process(float *buf, int n)
{
    if (!buf || n <= 0) return ESP_ERR_INVALID_ARG;
    /* 全局旁通: 直接返回 */
    if (s_fx.global_bypass) return ESP_OK;
    /* 无效果器: 直接返回 */
    if (s_fx.count == 0) return ESP_OK;

    ensure_lock();
    /* 取锁拷贝链指针, 处理期间不持锁 */
    fx_node_t *chain_snap[FX_CHAIN_MAX];
    int count_snap;
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    count_snap = s_fx.count;
    memcpy(chain_snap, s_fx.chain, sizeof(chain_snap));
    xSemaphoreGive(s_fx.lock);

    for (int k = 0; k < count_snap; k++) {
        fx_node_t *node = chain_snap[k];
        if (node && node->enabled && node->process) {
            node->process(node, buf, n);
        }
    }
    return ESP_OK;
}

/* ============================================================
 * 旁通控制
 * ============================================================ */
esp_err_t fx_chain_set_bypass(int bypass)
{
    s_fx.global_bypass = bypass ? 1 : 0;
    return ESP_OK;
}

int fx_chain_get_bypass(void)
{
    return s_fx.global_bypass;
}

/* ============================================================
 * 运行时链操作 (线程安全)
 * ============================================================ */
int fx_chain_add(fx_type_t type)
{
    if (type <= FX_NONE || type >= FX_TYPE_MAX) return -1;
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    if (s_fx.count >= FX_CHAIN_MAX) {
        xSemaphoreGive(s_fx.lock);
        ESP_LOGW(TAG, "链已满 (max %d)", FX_CHAIN_MAX);
        return -2;
    }
    fx_node_t *node = fx_node_create(type, s_fx.cfg.sample_rate > 0 ?
                                            s_fx.cfg.sample_rate : I2S_DEFAULT_SAMPLE_RATE);
    if (!node) {
        xSemaphoreGive(s_fx.lock);
        return -3;
    }
    int idx = s_fx.count++;
    s_fx.chain[idx] = node;
    xSemaphoreGive(s_fx.lock);
    return idx;
}

int fx_chain_insert(int index, fx_type_t type)
{
    if (type <= FX_NONE || type >= FX_TYPE_MAX) return -1;
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    if (s_fx.count >= FX_CHAIN_MAX) {
        xSemaphoreGive(s_fx.lock);
        return -2;
    }
    if (index < 0) index = 0;
    if (index > s_fx.count) index = s_fx.count;
    /* 后移 */
    for (int i = s_fx.count; i > index; i--)
        s_fx.chain[i] = s_fx.chain[i - 1];
    fx_node_t *node = fx_node_create(type, s_fx.cfg.sample_rate > 0 ?
                                            s_fx.cfg.sample_rate : I2S_DEFAULT_SAMPLE_RATE);
    if (!node) {
        /* 回滚 */
        for (int i = index; i < s_fx.count; i++)
            s_fx.chain[i] = s_fx.chain[i + 1];
        xSemaphoreGive(s_fx.lock);
        return -3;
    }
    s_fx.chain[index] = node;
    s_fx.count++;
    xSemaphoreGive(s_fx.lock);
    return index;
}

esp_err_t fx_chain_remove(int index)
{
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    if (index < 0 || index >= s_fx.count) {
        xSemaphoreGive(s_fx.lock);
        return ESP_ERR_INVALID_ARG;
    }
    fx_node_destroy(s_fx.chain[index]);
    /* 前移 */
    for (int i = index; i < s_fx.count - 1; i++)
        s_fx.chain[i] = s_fx.chain[i + 1];
    s_fx.chain[s_fx.count - 1] = NULL;
    s_fx.count--;
    xSemaphoreGive(s_fx.lock);
    return ESP_OK;
}

esp_err_t fx_chain_clear(void)
{
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    for (int i = 0; i < s_fx.count; i++) {
        if (s_fx.chain[i]) {
            fx_node_destroy(s_fx.chain[i]);
            s_fx.chain[i] = NULL;
        }
    }
    s_fx.count = 0;
    xSemaphoreGive(s_fx.lock);
    return ESP_OK;
}

esp_err_t fx_chain_set_enabled(int index, int enabled)
{
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    if (index < 0 || index >= s_fx.count) {
        xSemaphoreGive(s_fx.lock);
        return ESP_ERR_INVALID_ARG;
    }
    s_fx.chain[index]->enabled = enabled ? 1 : 0;
    xSemaphoreGive(s_fx.lock);
    return ESP_OK;
}

esp_err_t fx_chain_set_param(int index, const char *param, float value)
{
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    if (index < 0 || index >= s_fx.count) {
        xSemaphoreGive(s_fx.lock);
        return ESP_ERR_INVALID_ARG;
    }
    int rc = fx_node_set_param_by_name(s_fx.chain[index], param, value);
    xSemaphoreGive(s_fx.lock);
    return (rc == 0) ? ESP_OK : ESP_ERR_INVALID_ARG;
}

esp_err_t fx_chain_reset(void)
{
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    for (int i = 0; i < s_fx.count; i++) {
        if (s_fx.chain[i] && s_fx.chain[i]->reset)
            s_fx.chain[i]->reset(s_fx.chain[i]);
    }
    xSemaphoreGive(s_fx.lock);
    return ESP_OK;
}

/* ============================================================
 * 预设
 * ============================================================ */
typedef struct {
    const char *name;
    struct { fx_type_t type; const char *param; float value; } items[FX_CHAIN_MAX];
    int n_entries;
} fx_preset_t;

/* 内置预设: items[] 中连续相同 type 的项归为同一个效果器节点。
 * n_entries 是 items[] 中实际使用的条目数 (不是效果器数)。 */
static const fx_preset_t s_presets[] = {
    {
        "clean",  /* 清音: 压缩 + EQ + 轻混响 */
        {
            { FX_COMPRESSOR, "threshold", -24.0f },
            { FX_COMPRESSOR, "ratio",      3.0f  },
            { FX_EQ,         "low_db",     2.0f  },
            { FX_EQ,         "high_db",    1.0f  },
            { FX_REVERB,     "mix",        0.2f  },
            { FX_REVERB,     "room",       0.4f  },
        },
        6,
    },
    {
        "blues",  /* 布鲁斯: 压缩 + 轻失真 + 延迟 */
        {
            { FX_COMPRESSOR, "threshold", -20.0f },
            { FX_COMPRESSOR, "ratio",      4.0f  },
            { FX_DISTORTION, "drive",      0.3f  },
            { FX_DISTORTION, "mode",       0.0f  },  /* 软削波 */
            { FX_DELAY,      "time_ms",    300.0f },
            { FX_DELAY,      "mix",        0.25f },
        },
        6,
    },
    {
        "rock",  /* 摇滚: 压缩 + 强失真 + EQ */
        {
            { FX_COMPRESSOR, "threshold", -15.0f },
            { FX_COMPRESSOR, "ratio",      6.0f  },
            { FX_DISTORTION, "drive",      0.7f  },
            { FX_DISTORTION, "mode",       0.0f  },
            { FX_EQ,         "mid_db",     3.0f  },
            { FX_EQ,         "high_db",    2.0f  },
        },
        6,
    },
    {
        "metal",  /* 金属: 压缩 + 法兹 + EQ */
        {
            { FX_COMPRESSOR, "threshold", -10.0f },
            { FX_COMPRESSOR, "ratio",      8.0f  },
            { FX_DISTORTION, "drive",      0.9f  },
            { FX_DISTORTION, "mode",       2.0f  },  /* 法兹 */
            { FX_EQ,         "low_db",     4.0f  },
            { FX_EQ,         "high_db",    3.0f  },
        },
        6,
    },
    {
        "ambient",  /* 氛围: 长延迟 + 大混响 */
        {
            { FX_DELAY,      "time_ms",    450.0f },
            { FX_DELAY,      "feedback",   0.6f  },
            { FX_DELAY,      "mix",        0.4f  },
            { FX_REVERB,     "room",       0.9f  },
            { FX_REVERB,     "mix",        0.5f  },
        },
        5,
    },
};
#define N_PRESETS  (sizeof(s_presets) / sizeof(s_presets[0]))

esp_err_t fx_chain_load_preset(const char *name)
{
    if (!name) return ESP_ERR_INVALID_ARG;
    int found = -1;
    for (int i = 0; i < (int)N_PRESETS; i++) {
        if (!strcmp(s_presets[i].name, name)) { found = i; break; }
    }
    if (found < 0) {
        ESP_LOGW(TAG, "未知预设: %s", name);
        return ESP_ERR_NOT_FOUND;
    }
    const fx_preset_t *p = &s_presets[found];
    fx_chain_clear();
    /* 遍历条目, 连续相同 type 的参数归为同一个效果器节点 */
    int i = 0;
    int node_count = 0;
    while (i < p->n_entries) {
        fx_type_t t = p->items[i].type;
        int idx = fx_chain_add(t);
        if (idx < 0) return ESP_FAIL;
        node_count++;
        /* 应用所有属于这个效果器的参数 (连续相同 type) */
        while (i < p->n_entries && p->items[i].type == t) {
            fx_chain_set_param(idx, p->items[i].param, p->items[i].value);
            i++;
        }
    }
    ESP_LOGI(TAG, "已加载预设: %s (%d 个效果器)", name, node_count);
    return ESP_OK;
}

void fx_chain_list_presets(char *buf, size_t n)
{
    if (!buf || n == 0) return;
    buf[0] = 0;
    for (int i = 0; i < (int)N_PRESETS; i++) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%s%s", (i > 0) ? " | " : "", s_presets[i].name);
        strncat(buf, tmp, n - strlen(buf) - 1);
    }
}

/* ============================================================
 * 状态与打印
 * ============================================================ */
void fx_chain_status(char *buf, size_t n)
{
    if (!buf || n == 0) return;
    if (!s_fx.running) {
        snprintf(buf, n, "效果器链: 停止 (链长 %d, %s)",
                 s_fx.count, s_fx.global_bypass ? "全局旁通" : "启用");
        return;
    }
    int64_t avg_us = s_fx.frames > 0 ? s_fx.sum_us / s_fx.frames : 0;
    double rt_factor = (avg_us > 0) ?
        (1000.0 * s_fx.cfg.hop_size / s_fx.cfg.sample_rate) / (avg_us / 1000.0) : 0.0;
    snprintf(buf, n,
             "效果器链: 运行中 %s 链长=%d  %" PRIu32 "帧 avg=%" PRId64 "us(%.1fxRT) under=%d over=%d",
             s_fx.global_bypass ? "[旁通]" : "",
             s_fx.count, s_fx.frames, avg_us, rt_factor, s_fx.underruns, s_fx.overruns);
}

void fx_chain_print(void)
{
    printf("效果器链 (共 %d 个, %s):\n", s_fx.count,
           s_fx.global_bypass ? "全局旁通" : "启用");
    ensure_lock();
    xSemaphoreTake(s_fx.lock, portMAX_DELAY);
    for (int i = 0; i < s_fx.count; i++) {
        fx_node_t *node = s_fx.chain[i];
        if (!node) continue;
        printf("  [%d] %s %s\n", i, fx_type_name(node->type),
               node->enabled ? "(启用)" : "(旁通)");
        for (int j = 0; j < node->num_params; j++) {
            printf("       %s = %.3f  [%.3f~%.3f]\n",
                   node->params[j].name, node->params[j].value,
                   node->params[j].min_val, node->params[j].max_val);
        }
    }
    xSemaphoreGive(s_fx.lock);
    if (s_fx.count == 0) {
        printf("  (空链, 使用 'fx add <类型>' 添加效果器)\n");
    }
}
