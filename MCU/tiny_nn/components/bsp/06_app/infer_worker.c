#include "infer_worker.h"
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "infer_worker";

#define INFER_QUEUE_LEN     4
#define INFER_TASK_STACK    16384
#define INFER_TASK_PRIO     5
#define INFER_TASK_CORE     1     /* 推理固定 core1, CLI 在 core0 */

static QueueHandle_t     s_queue = NULL;
static SemaphoreHandle_t s_lock  = NULL;
static TaskHandle_t      s_task  = NULL;

/* 共享状态 (s_lock 保护) */
static struct {
    int   busy;
    char  cur[24];
    int   done;
    int   last_rc;
    int   last_ms;
    char  last_name[24];
} s_st;

static void st_set_busy(const char *name)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_st.busy = 1;
    strncpy(s_st.cur, name, sizeof(s_st.cur) - 1);
    s_st.cur[sizeof(s_st.cur) - 1] = '\0';
    xSemaphoreGive(s_lock);
}

static void st_set_done(const char *name, int rc, int ms)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_st.busy = 0;
    s_st.cur[0] = '\0';
    s_st.done++;
    s_st.last_rc = rc;
    s_st.last_ms = ms;
    strncpy(s_st.last_name, name, sizeof(s_st.last_name) - 1);
    s_st.last_name[sizeof(s_st.last_name) - 1] = '\0';
    xSemaphoreGive(s_lock);
}

static void ensure_parent_dir(const char *path)
{
    /* 取最后一个 '/' 之前的目录, mkdir (单层, 已存在则忽略) */
    const char *slash = strrchr(path, '/');
    if (!slash || slash == path) return;
    char dir[112];
    size_t len = (size_t)(slash - path);
    if (len >= sizeof(dir)) return;
    memcpy(dir, path, len);
    dir[len] = '\0';
    mkdir(dir, 0777);
}

static void worker_task(void *arg)
{
    (void)arg;
    infer_job_t job;
    for (;;) {
        if (xQueueReceive(s_queue, &job, portMAX_DELAY) != pdTRUE) continue;

        ESP_LOGI(TAG, "开始推理: %s  %s -> %s (pitch %+.1f, gain %+.1fdB, clip %d)",
                 job.instrument, job.in_path, job.out_path,
                 job.opt.pitch_semitones, job.opt.gain_db, (int)job.opt.clip_mode);
        st_set_busy(job.instrument);
        ensure_parent_dir(job.out_path);

        int64_t t0 = esp_timer_get_time();
        esp_err_t rc = audio_xform_file(job.in_path, job.out_path, job.instrument, &job.opt);
        int ms = (int)((esp_timer_get_time() - t0) / 1000);

        st_set_done(job.instrument, (int)rc, ms);
        if (rc == ESP_OK)
            ESP_LOGI(TAG, "完成: %s -> %s (%dms)", job.instrument, job.out_path, ms);
        else if (rc == ESP_ERR_NOT_FOUND)
            ESP_LOGW(TAG, "跳过: 模型不含乐器 '%s'", job.instrument);
        else
            ESP_LOGE(TAG, "推理失败: %s rc=%s", job.instrument, esp_err_to_name(rc));
    }
}

esp_err_t infer_worker_start(void)
{
    if (s_task) return ESP_OK;
    s_lock = xSemaphoreCreateMutex();
    s_queue = xQueueCreate(INFER_QUEUE_LEN, sizeof(infer_job_t));
    if (!s_lock || !s_queue) return ESP_ERR_NO_MEM;
    memset(&s_st, 0, sizeof(s_st));
    BaseType_t ok = xTaskCreatePinnedToCore(worker_task, "infer", INFER_TASK_STACK,
                                            NULL, INFER_TASK_PRIO, &s_task, INFER_TASK_CORE);
    if (ok != pdPASS) { s_task = NULL; return ESP_ERR_NO_MEM; }
    ESP_LOGI(TAG, "worker 就绪 (core%d, 队列深度 %d)", INFER_TASK_CORE, INFER_QUEUE_LEN);
    return ESP_OK;
}

esp_err_t infer_worker_submit(const infer_job_t *job)
{
    if (!s_queue || !job) return ESP_ERR_INVALID_STATE;
    if (xQueueSend(s_queue, job, 0) != pdTRUE) return ESP_ERR_NO_MEM;
    return ESP_OK;
}

int infer_worker_busy(void)
{
    if (!s_lock) return 0;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    int b = s_st.busy;
    xSemaphoreGive(s_lock);
    return b;
}

void infer_worker_status(char *buf, size_t n)
{
    if (!buf || n == 0) return;
    if (!s_lock) { snprintf(buf, n, "worker 未启动"); return; }
    UBaseType_t waiting = s_queue ? uxQueueMessagesWaiting(s_queue) : 0;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    if (s_st.busy)
        snprintf(buf, n, "BUSY 正在推理 '%s' | 队列等待 %u | 已完成 %d",
                 s_st.cur, (unsigned)waiting, s_st.done);
    else if (s_st.done > 0)
        snprintf(buf, n, "IDLE | 队列等待 %u | 已完成 %d | 上次: %s rc=%d %dms",
                 (unsigned)waiting, s_st.done, s_st.last_name, s_st.last_rc, s_st.last_ms);
    else
        snprintf(buf, n, "IDLE | 队列等待 %u | 尚未执行作业", (unsigned)waiting);
    xSemaphoreGive(s_lock);
}
