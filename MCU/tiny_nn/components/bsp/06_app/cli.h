/* 命令行系统: esp_console REPL (UART, 运行在 core0), 触发 core1 推理 worker.
 * 推理进行时命令行仍可用 (查询状态/排队新作业). */
#ifndef CLI_H
#define CLI_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 启动 REPL 并注册命令 (model / infer / status / ls / sdinfo). 需先启动 infer_worker. */
esp_err_t cli_start(void);

#ifdef __cplusplus
}
#endif
#endif
