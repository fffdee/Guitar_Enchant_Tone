#include "bnn_utils/bnn_log.h"
#include "bnn_utils/bnn_config.h"

#include <stdarg.h>
#include <stdio.h>

/*
 * 日志实现 (与硬件解耦) + 可移植计时钩子.
 *  - bnn_log_emit 把一行格式化到栈上小缓冲, 再交给当前 sink;
 *  - 默认 sink 由 BNN_LOG_USE_STDIO 决定是否写 stdout/stderr;
 *  - MCU 上把 sink 重定向到 UART 即可, 无需改动任何调用点.
 */

#ifndef BNN_LOG_LINE_MAX
#define BNN_LOG_LINE_MAX 256
#endif

static void default_sink(int level, const char *msg, void *user) {
    (void)user;
#if BNN_LOG_USE_STDIO
    FILE *out = (level <= 1) ? stderr : stdout;
    fputs(msg, out);
    fputc('\n', out);
#else
    (void)level;
    (void)msg;
#endif
}

static bnn_log_sink_fn g_sink = default_sink;
static void           *g_sink_user = 0;

void bnn_log_set_sink(bnn_log_sink_fn fn, void *user) {
    g_sink = fn ? fn : default_sink;
    g_sink_user = user;
}

void bnn_log_emit(int level, const char *fmt, ...) {
    if (!g_sink) return;
    char line[BNN_LOG_LINE_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    line[sizeof(line) - 1] = '\0';
    g_sink(level, line, g_sink_user);
}

/* ----- 可移植计时钩子 (host: clock_gettime; MCU: DWT/timer) ----- */
static bnn_time_us_fn g_time_us = 0;

void     bnn_time_set_source(bnn_time_us_fn fn) { g_time_us = fn; }
uint32_t bnn_time_us(void) { return g_time_us ? g_time_us() : 0u; }
