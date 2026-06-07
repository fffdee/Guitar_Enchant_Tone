#ifndef BNN_CONFIG_H
#define BNN_CONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 全局编译开关与可移植钩子 (与硬件/部署形态解耦).
 *
 *  BNN_ENABLE_TRAINING
 *    1 = 编入反向/优化器/损失/数据集 (host 训练);
 *    0 = 纯推理部署, 裁掉训练件以减小 flash (MCU 默认).
 *    新增的推理层(conv1d/film/embedding)据此决定是否提供 backward.
 *
 *  BNN_INFERENCE_ONLY  等价 (BNN_ENABLE_TRAINING==0) 的语义别名.
 */
#ifndef BNN_ENABLE_TRAINING
#  if defined(BNN_PLATFORM_MCU)
#    define BNN_ENABLE_TRAINING 0
#  else
#    define BNN_ENABLE_TRAINING 1
#  endif
#endif

#if BNN_ENABLE_TRAINING
#  define BNN_INFERENCE_ONLY 0
#else
#  define BNN_INFERENCE_ONLY 1
#endif

/*
 * BNN_ENABLE_FILE_IO
 *   1 = 编入基于 <stdio.h> FILE* 的权重存取 (host 有文件系统);
 *   0 = 裁掉 fopen/fread/fwrite, 彻底不依赖文件系统 (MCU 默认).
 *   关闭时仍可用 bnn_graph_load_weights_mem 从内存/Flash 数组加载.
 */
#ifndef BNN_ENABLE_FILE_IO
#  if defined(BNN_PLATFORM_MCU)
#    define BNN_ENABLE_FILE_IO 0
#  else
#    define BNN_ENABLE_FILE_IO 1
#  endif
#endif

/*
 * BNN_LOG_USE_STDIO
 *   1 = 默认日志后端走 stdout/stderr (host);
 *   0 = 不引用 stdio, 未设置 sink 时日志为 no-op (MCU 默认, 避免强依赖标准输出).
 *   任何平台均可用 bnn_log_set_sink() 把日志重定向到 UART / SEGGER-RTT 等.
 */
#ifndef BNN_LOG_USE_STDIO
#  if defined(BNN_PLATFORM_MCU)
#    define BNN_LOG_USE_STDIO 0
#  else
#    define BNN_LOG_USE_STDIO 1
#  endif
#endif

/*
 * 可移植微秒计时钩子: 用于延迟剖析 (替代 host 专用的 clock()).
 *  - 默认返回 0 (未设置时);
 *  - host:  bnn_time_set_source(基于 clock_gettime 的函数);
 *  - MCU :  bnn_time_set_source(基于 DWT CYCCNT / 硬件 timer 的函数).
 */
typedef uint32_t (*bnn_time_us_fn)(void);

void     bnn_time_set_source(bnn_time_us_fn fn);
uint32_t bnn_time_us(void);

#ifdef __cplusplus
}
#endif
#endif
