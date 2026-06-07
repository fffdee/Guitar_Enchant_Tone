#ifndef BNN_LOG_H
#define BNN_LOG_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BNN_LOG_LEVEL
#define BNN_LOG_LEVEL 2   /* 0:none 1:err 2:info 3:debug */
#endif

/*
 * 日志后端与硬件解耦:
 *  - 所有日志经 bnn_log_emit() 汇聚, 格式化后交给"日志 sink";
 *  - 默认 sink 在 BNN_LOG_USE_STDIO=1 时写 stdout/stderr, =0 时为 no-op;
 *  - 任意平台可调用 bnn_log_set_sink() 重定向到 UART / RTT / 文件 / ringbuffer.
 *  - 关闭的日志级别在编译期消除 (不评估参数, 零开销).
 *
 * level: 1=err 2=info 3=debug (与 BNN_LOG_LEVEL 对应).
 */
typedef void (*bnn_log_sink_fn)(int level, const char *msg, void *user);

void bnn_log_set_sink(bnn_log_sink_fn fn, void *user);
void bnn_log_emit(int level, const char *fmt, ...);

#if BNN_LOG_LEVEL >= 1
#define BNN_LOGE(fmt, ...) bnn_log_emit(1, "[BNN-E] " fmt, ##__VA_ARGS__)
#else
#define BNN_LOGE(fmt, ...) ((void)0)
#endif

#if BNN_LOG_LEVEL >= 2
#define BNN_LOGI(fmt, ...) bnn_log_emit(2, "[BNN-I] " fmt, ##__VA_ARGS__)
#else
#define BNN_LOGI(fmt, ...) ((void)0)
#endif

#if BNN_LOG_LEVEL >= 3
#define BNN_LOGD(fmt, ...) bnn_log_emit(3, "[BNN-D] " fmt, ##__VA_ARGS__)
#else
#define BNN_LOGD(fmt, ...) ((void)0)
#endif

#define BNN_CHECK(cond, msg) do { if(!(cond)){ BNN_LOGE("CHECK failed: %s | %s", #cond, msg); } } while(0)
#define BNN_CHECK_RET(cond, ret) do { if(!(cond)){ BNN_LOGE("CHECK failed: %s", #cond); return (ret); } } while(0)

#ifdef __cplusplus
}
#endif

#endif
