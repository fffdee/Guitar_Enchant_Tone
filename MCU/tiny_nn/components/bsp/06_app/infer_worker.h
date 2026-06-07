/* 推理 worker: 固定在 core1 的后台任务, 通过队列接收作业并执行音色转换.
 * CLI(core0) 只负责入队, 推理期间命令行保持可用 (双核解耦). */
#ifndef INFER_WORKER_H
#define INFER_WORKER_H

#include "esp_err.h"
#include <stddef.h>
#include "audio_xform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 一个推理作业 (CLI 组装, worker 执行) */
typedef struct {
    char              instrument[24];
    char              in_path[112];
    char              out_path[112];
    audio_xform_opt_t opt;
} infer_job_t;

/* 启动 worker (创建队列 + core1 任务). 幂等. */
esp_err_t infer_worker_start(void);

/* 入队一个作业 (非阻塞). 队列满返回 ESP_ERR_NO_MEM. */
esp_err_t infer_worker_submit(const infer_job_t *job);

/* 当前是否在推理 */
int infer_worker_busy(void);

/* 把可读状态写入 buf (busy/idle + 当前作业 + 已完成数 + 上次耗时/结果) */
void infer_worker_status(char *buf, size_t n);

#ifdef __cplusplus
}
#endif
#endif
