/* I2S 实时推理流水线实现。
 *
 * 数据流:
 *   I2S RX (16bit) -> int16 转 float -> 累积到 hop 帧 -> bnn_masknet 推理
 *   -> 后处理(变调/增益/削波) -> float 转 int16 -> I2S TX
 *
 * 帧长: 与模型 hop 一致 (默认 240 采样 @48kHz = 5ms/帧)。
 * 延迟: ~2 帧 (输入缓冲 1 帧 + 推理 1 帧) ≈ 10ms, 满足 <15ms 要求。
 */
#include "i2s_xform.h"
#include "i2s_driver.h"
#include "audio_xform.h"
#include "audio_post.h"
#include "model_store.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "bnn_model/bnn_masknet.h"

/* 外部引用 (audio_xform.c 中定义) */
extern bnn_masknet_t *s_net;
extern bnn_mask_cfg_t s_cfg;
extern model_pkg_t    s_pkg;

static const char *TAG = "i2s_xform";

#define I2S_TASK_STACK   8192
#define I2S_TASK_PRIO    7       /* 高于 infer_worker(5), 保证实时性 */
#define I2S_TASK_CORE    1       /* 与 infer_worker 共享 core1, 但互斥运行 */

/* 运行时状态 */
static struct {
    TaskHandle_t      task;
    SemaphoreHandle_t lock;
    i2s_xform_cfg_t  cfg;
    volatile int      running;
    volatile int      stop_req;
    /* 统计 */
    uint32_t          frames;
    int64_t           sum_us;
    int64_t           min_us;
    int64_t           max_us;
    int               underruns;   /* I2S 读欠载次数 */
    int               overruns;    /* I2S 写溢出次数 */
} s_rt;

/* int16 <-> float 转换 */
static inline float i16_to_f(int16_t s) { return (float)s / 32768.0f; }
static inline int16_t f_to_i16(float f) {
    if (f >= 1.0f)  return 32767;
    if (f <= -1.0f) return -32768;
    return (int16_t)(int32_t)(f * 32768.0f);
}

static void rt_task(void *arg)
{
    (void)arg;
    int hop = s_cfg.hop;                       /* 每帧采样数 (240) */
    int sr  = s_cfg.sample_rate;

    /* 分配缓冲 */
    int16_t *i2s_buf = (int16_t *)heap_caps_malloc((size_t)hop * sizeof(int16_t),
                                                    MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    float   *in_buf  = (float *)heap_caps_malloc((size_t)hop * sizeof(float),
                                                  MALLOC_CAP_SPIRAM);
    float   *out_buf = (float *)heap_caps_malloc((size_t)hop * sizeof(float),
                                                  MALLOC_CAP_SPIRAM);
    if (!i2s_buf || !in_buf || !out_buf) {
        ESP_LOGE(TAG, "缓冲分配失败");
        goto cleanup;
    }

    /* 设置乐器 */
    int inst_id = model_store_instrument_id(&s_pkg, s_rt.cfg.instrument);
    if (inst_id < 0) {
        ESP_LOGE(TAG, "乐器 '%s' 不在模型中", s_rt.cfg.instrument);
        goto cleanup;
    }
    bnn_masknet_set_instrument(s_net, inst_id);
    bnn_masknet_set_add_noise(s_net, s_rt.cfg.add_noise);
    bnn_masknet_reset(s_net);

    /* 启动 I2S 收发 */
    esp_err_t err = i2s_driver_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S 启动失败: %s", esp_err_to_name(err));
        goto cleanup;
    }

    ESP_LOGI(TAG, "实时推理启动: %s  hop=%d  sr=%d", s_rt.cfg.instrument, hop, sr);
    s_rt.running = 1;

    /* 主循环: 读一帧 -> 推理 -> 写一帧 */
    while (!s_rt.stop_req) {
        /* 1. 从 I2S RX 读取一帧 (hop 个 16-bit 采样) */
        int got = i2s_driver_read(i2s_buf, hop, 100);
        if (got < hop) {
            s_rt.underruns++;
            /* 不足一帧时补零 */
            for (int i = (got > 0 ? got : 0); i < hop; i++)
                i2s_buf[i] = 0;
        }

        /* 2. int16 -> float */
        for (int i = 0; i < hop; i++)
            in_buf[i] = i16_to_f(i2s_buf[i]);

        /* 3. 推理 */
        int64_t t0 = esp_timer_get_time();
        int out_n = 0;
        int rc = bnn_masknet_process_audio(s_net, in_buf, hop, out_buf, &out_n);
        int64_t dt_us = esp_timer_get_time() - t0;

        if (rc != 0) {
            ESP_LOGW(TAG, "推理失败 rc=%d", rc);
            continue;
        }

        /* 4. 后处理 (变调/增益/削波) */
        xSemaphoreTake(s_rt.lock, portMAX_DELAY);
        i2s_xform_cfg_t cfg = s_rt.cfg;
        xSemaphoreGive(s_rt.lock);

        if (cfg.pitch_semitones != 0.0f)
            audio_post_pitch_shift(out_buf, out_n, sr, cfg.pitch_semitones,
                                   s_cfg.n_fft, s_cfg.hop);
        if (cfg.gain_db != 0.0f || cfg.clip_mode != AUDIO_CLIP_LIMIT)
            audio_post_gain_clip(out_buf, out_n, cfg.gain_db, cfg.clip_mode);

        /* 5. float -> int16 */
        int write_n = (out_n > hop) ? hop : out_n;
        for (int i = 0; i < write_n; i++)
            i2s_buf[i] = f_to_i16(out_buf[i]);
        /* 输出不足时补零 */
        for (int i = write_n; i < hop; i++)
            i2s_buf[i] = 0;

        /* 6. 写入 I2S TX */
        int written = i2s_driver_write(i2s_buf, hop, 10);
        if (written < hop)
            s_rt.overruns++;

        /* 7. 更新统计 */
        s_rt.frames++;
        s_rt.sum_us += dt_us;
        if (dt_us < s_rt.min_us) s_rt.min_us = dt_us;
        if (dt_us > s_rt.max_us) s_rt.max_us = dt_us;
    }

    /* 停止 I2S */
    i2s_driver_stop();
    s_rt.running = 0;
    ESP_LOGI(TAG, "实时推理已停止: %u帧  avg=%lldus  underruns=%d  overruns=%d",
             s_rt.frames, s_rt.frames > 0 ? s_rt.sum_us / s_rt.frames : 0,
             s_rt.underruns, s_rt.overruns);

cleanup:
    if (i2s_buf) heap_caps_free(i2s_buf);
    if (in_buf)  heap_caps_free(in_buf);
    if (out_buf) heap_caps_free(out_buf);
    s_rt.running = 0;
    s_rt.task = NULL;
    vTaskDelete(NULL);
}

esp_err_t i2s_xform_start(const i2s_xform_cfg_t *cfg)
{
    if (!cfg) return ESP_ERR_INVALID_ARG;
    if (s_rt.running) {
        ESP_LOGW(TAG, "已在运行, 请先停止");
        return ESP_ERR_INVALID_STATE;
    }
    if (!audio_xform_loaded()) {
        ESP_LOGE(TAG, "模型未加载");
        return ESP_ERR_INVALID_STATE;
    }
    if (!i2s_driver_is_ready()) {
        ESP_LOGE(TAG, "I2S 未初始化");
        return ESP_ERR_INVALID_STATE;
    }

    /* 重置状态 */
    memset(&s_rt, 0, sizeof(s_rt));
    s_rt.min_us = INT64_MAX;
    s_rt.lock = xSemaphoreCreateMutex();
    if (!s_rt.lock) return ESP_ERR_NO_MEM;
    memcpy(&s_rt.cfg, cfg, sizeof(s_rt.cfg));

    BaseType_t ok = xTaskCreatePinnedToCore(rt_task, "i2s_xform", I2S_TASK_STACK,
                                            NULL, I2S_TASK_PRIO, &s_rt.task, I2S_TASK_CORE);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "任务创建失败");
        vSemaphoreDelete(s_rt.lock);
        s_rt.lock = NULL;
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t i2s_xform_stop(void)
{
    if (!s_rt.running && !s_rt.task) return ESP_OK;
    s_rt.stop_req = 1;
    /* 等待任务退出 (最多 2 秒) */
    for (int i = 0; i < 200 && s_rt.running; i++)
        vTaskDelay(pdMS_TO_TICKS(10));
    if (s_rt.lock) { vSemaphoreDelete(s_rt.lock); s_rt.lock = NULL; }
    return ESP_OK;
}

int i2s_xform_running(void)
{
    return s_rt.running;
}

esp_err_t i2s_xform_set_instrument(const char *instrument)
{
    if (!s_rt.running || !s_net) return ESP_ERR_INVALID_STATE;
    int id = model_store_instrument_id(&s_pkg, instrument);
    if (id < 0) return ESP_ERR_NOT_FOUND;
    bnn_masknet_set_instrument(s_net, id);
    xSemaphoreTake(s_rt.lock, portMAX_DELAY);
    strncpy(s_rt.cfg.instrument, instrument, sizeof(s_rt.cfg.instrument) - 1);
    xSemaphoreGive(s_rt.lock);
    return ESP_OK;
}

esp_err_t i2s_xform_set_post(float pitch_semitones, float gain_db,
                              audio_clip_mode_t clip_mode, int add_noise)
{
    if (!s_rt.running || !s_net) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(s_rt.lock, portMAX_DELAY);
    s_rt.cfg.pitch_semitones = pitch_semitones;
    s_rt.cfg.gain_db = gain_db;
    s_rt.cfg.clip_mode = clip_mode;
    s_rt.cfg.add_noise = add_noise;
    xSemaphoreGive(s_rt.lock);
    bnn_masknet_set_add_noise(s_net, add_noise);
    return ESP_OK;
}

void i2s_xform_status(char *buf, size_t n)
{
    if (!buf || n == 0) return;
    if (!s_rt.running) {
        snprintf(buf, n, "I2S 实时推理: 停止");
        return;
    }
    int64_t avg_us = s_rt.frames > 0 ? s_rt.sum_us / s_rt.frames : 0;
    double rt_factor = (avg_us > 0) ?
        (1000.0 * s_cfg.hop / s_cfg.sample_rate) / (avg_us / 1000.0) : 0.0;
    snprintf(buf, n,
             "I2S 实时: %s  %u帧  avg=%lldus(%.1fxRT)  min=%lldus  max=%lldus  under=%d  over=%d",
             s_rt.cfg.instrument, s_rt.frames, avg_us, rt_factor,
             s_rt.min_us < INT64_MAX ? s_rt.min_us : 0,
             s_rt.max_us, s_rt.underruns, s_rt.overruns);
}
