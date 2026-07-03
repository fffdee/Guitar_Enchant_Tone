/* bnn_perf.h — 性能优化开关集中配置。
 *
 * 所有优化均通过宏开关控制, 方便回退与 A/B 对比。
 * 在 sdkconfig 或 CMakeLists 中定义这些宏即可启用对应优化。
 *
 * 优化项 (按收益排序):
 *   BNN_PERF_DUAL_CORE   方案1: Conv1d 通道分割双核并行 (~1.8x)
 *   BNN_PERF_WEIGHT_PREFETCH  权重预取到 SRAM (~1.2x 额外)
 *   BNN_PERF_FUSE_BIAS_RELU   算子融合 Conv+bias+ReLU (~1.1x 额外)
 *
 * 组合使用:
 *   全开: 3 个宏都定义, 累计 ~2.4x (F32)
 *   仅双核: 只定义 BNN_PERF_DUAL_CORE, ~1.8x
 *   全关: 都不定义, 回退到原始单核 F32 路径
 *
 * 注意:
 *   - BNN_PERF_DUAL_CORE 需要 FreeRTOS 双核支持 (ESP32-P4 标配)
 *   - BNN_PERF_WEIGHT_PREFETCH 需要足够 SRAM 空间 (~50KB/层)
 *   - BNN_PERF_FUSE_BIAS_RELU 需要修改 GEMM epilogue, 与 INT8 路径互斥
 */
#ifndef BNN_PERF_H
#define BNN_PERF_H

/* ============================================================
 * 方案1: Conv1d 通道分割双核并行
 * ============================================================
 * 原理: 每个 Conv1d 的输出通道 (Cout) 对半分, core1 算前半,
 *       core0 算后半, 层间用 task notification 同步。
 *
 * 数据流:
 *   core1: im2col -> GEMM(W[0:Cout/2]) -> 写 Y[0:Cout/2]
 *   core0:                  GEMM(W[Cout/2:Cout]) -> 写 Y[Cout/2:Cout]
 *   同步: 层结束时 barrier (xTaskNotifyWait)
 *
 * 收益: ~1.8x (扣除同步开销)
 * 限制: 仅 F32 路径有效 (INT8 走 esp_nn_conv_s8 无法分割)
 */
#ifndef BNN_PERF_DUAL_CORE
#define BNN_PERF_DUAL_CORE 0
#endif

/* ============================================================
 * 权重预取到 SRAM
 * ============================================================
 * 原理: 推理前把当前层权重从 PSRAM 拷贝到 SRAM, 消除 PSRAM
 *       带宽竞争 (双核同时读 PSRAM 会降速)。
 *
 * 实现: 双缓冲, 后台 DMA 搬运下一层权重到 SRAM 空闲区。
 *       当前层用 SRAM 副本做 GEMM。
 *
 * 收益: ~1.2x 额外 (在双核基础上)
 * 限制: SRAM 空间有限, 单层权重约 50KB, 需 ~100KB 双缓冲
 */
#ifndef BNN_PERF_WEIGHT_PREFETCH
#define BNN_PERF_WEIGHT_PREFETCH 0
#endif

/* ============================================================
 * 算子融合: Conv + bias + ReLU
 * ============================================================
 * 原理: 在 GEMM 的 epilogue 中直接加 bias 并应用 ReLU, 消除
 *       中间结果的 PSRAM 读写往返。
 *
 * 实现: 自定义 gemm_nt_fused 函数, 在 dotprod 结果上原地加 bias
 *       并 clamp 到 [0, +inf)。
 *
 * 收益: ~1.1x 额外
 * 限制: 仅对 Conv1d 后接 ReLU 的层有效 (c1/c2/c3);
 *       head 层后接 sigmoid/tanh/softplus 不适用;
 *       与 INT8 路径互斥 (INT8 内部已融合)。
 */
#ifndef BNN_PERF_FUSE_BIAS_RELU
#define BNN_PERF_FUSE_BIAS_RELU 0
#endif

/* ============================================================
 * 运行时查询 (供调试/日志)
 * ============================================================ */
#if BNN_PERF_DUAL_CORE
#define BNN_PERF_DUAL_CORE_STR "ON"
#else
#define BNN_PERF_DUAL_CORE_STR "OFF"
#endif

#if BNN_PERF_WEIGHT_PREFETCH
#define BNN_PERF_WEIGHT_PREFETCH_STR "ON"
#else
#define BNN_PERF_WEIGHT_PREFETCH_STR "OFF"
#endif

#if BNN_PERF_FUSE_BIAS_RELU
#define BNN_PERF_FUSE_BIAS_RELU_STR "ON"
#else
#define BNN_PERF_FUSE_BIAS_RELU_STR "OFF"
#endif

/* 打印当前优化配置 (用于启动日志) */
#define BNN_PERF_PRINT_CONFIG()                                              \
    do {                                                                     \
        printf("[bnn_perf] DUAL_CORE=%s WEIGHT_PREFETCH=%s FUSE_BIAS_RELU=%s\n", \
               BNN_PERF_DUAL_CORE_STR, BNN_PERF_WEIGHT_PREFETCH_STR,         \
               BNN_PERF_FUSE_BIAS_RELU_STR);                                 \
    } while (0)

#endif /* BNN_PERF_H */
