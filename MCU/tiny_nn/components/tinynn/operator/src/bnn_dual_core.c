/* bnn_dual_core.c — 双核并行 GEMM 辅助 (方案1: 通道分割)。
 *
 * 仅在 BNN_PERF_DUAL_CORE=1 时编译。
 *
 * 工作原理:
 *   主核 (core1) 调用 bnn_dual_core_gemm_nt_split() 时:
 *   1. 把 GEMM 的 M 维 (输出通道 Cout) 对半分
 *   2. 通过 task notification 唤醒 core0 上的辅助任务
 *   3. core0 计算后半部分 [Cout/2 : Cout]
 *   4. 主核同时计算前半部分 [0 : Cout/2]
 *   5. 主核完成后等待 core0 完成 (再次 notification)
 *
 * 同步开销: 2 次 task notification ≈ 5-10μs, 可忽略。
 *
 * 注意:
 *   - 仅 F32 路径有效, INT8 走 esp_nn_conv_s8 无法分割
 *   - GEMM 后端 (dsps_dotprod_f32_arp4) 是线程安全的 (PIE 无状态)
 *   - workspace 需要各自独立, 不共享 arena
 */
#include "bnn_utils/bnn_perf.h"

#if BNN_PERF_DUAL_CORE

#include "bnn_op/bnn_op.h"
#include "bnn_utils/bnn_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

static const char *TAG = "bnn_dual";

/* 辅助任务参数 (一次 GEMM 分配) */
typedef struct {
    const float *A;       /* 权重 [M, K] (本核处理 M_half 行) */
    const float *B;       /* col_T [N, K] */
    const float *bias;    /* bias [M] (本核处理 M_half 个) */
    float       *C;       /* 输出 [M, N] (本核写入 M_half 行) */
    int          M_half;  /* 本核处理的行数 */
    int          N;
    int          K;
    int          acc;
    volatile int done;    /* 完成标志 */
} gemm_job_t;

/* 全局状态 */
static struct {
    TaskHandle_t      helper_task;   /* core0 辅助任务 */
    SemaphoreHandle_t job_sem;       /* 唤醒辅助任务 */
    SemaphoreHandle_t done_sem;      /* 完成信号 */
    gemm_job_t        job;           /* 当前作业 */
    int               initialized;
} s_dc;

/* core0 辅助任务: 等待作业, 执行 GEMM, 通知完成 */
static void helper_task(void *arg)
{
    (void)arg;
    while (1) {
        /* 等待主核派发作业 */
        xSemaphoreTake(s_dc.job_sem, portMAX_DELAY);
        if (!s_dc.job.A) continue;  /* 空作业 (不应发生) */

        /* 执行后半部分 GEMM */
        const bnn_op_backend_t *op = BNN_OP();
        /* 注意: 这里 M 是 M_half, A 指向后半部分起始 */
        op->gemm_nt(s_dc.job.A, s_dc.job.B, s_dc.job.bias,
                    s_dc.job.C, s_dc.job.M_half, s_dc.job.N,
                    s_dc.job.K, s_dc.job.acc);

        /* 通知主核完成 */
        s_dc.job.done = 1;
        xSemaphoreGive(s_dc.done_sem);
    }
}

/* 初始化双核并行 (懒初始化, 首次调用时触发) */
static int ensure_init(void)
{
    if (s_dc.initialized) return 0;
    s_dc.job_sem  = xSemaphoreCreateBinary();
    s_dc.done_sem = xSemaphoreCreateBinary();
    if (!s_dc.job_sem || !s_dc.done_sem) {
        BNN_LOGE("dual_core: 信号量创建失败");
        return -1;
    }
    BaseType_t ok = xTaskCreatePinnedToCore(helper_task, "bnn_dc", 4096,
                                            NULL, 5, &s_dc.helper_task, 0);
    if (ok != pdPASS) {
        BNN_LOGE("dual_core: 辅助任务创建失败");
        return -1;
    }
    s_dc.initialized = 1;
    BNN_LOGI("dual_core: 辅助任务已启动 (core0)");
    return 0;
}

/* 分割 GEMM: 主核算前半, core0 算后半。
 *
 * 参数同 op->gemm_nt, 但内部把 M 对半分。
 * A[M,K], B[N,K], C[M,N], bias[M]
 */
void bnn_dual_core_gemm_nt_split(const float *A, const float *B,
                                  const float *bias, float *C,
                                  int M, int N, int K, int acc)
{
    if (ensure_init() != 0) {
        /* 初始化失败, 回退单核 */
        BNN_OP()->gemm_nt(A, B, bias, C, M, N, K, acc);
        return;
    }

    int M_half = M / 2;
    int M_rest = M - M_half;

    /* 派发后半部分给 core0 */
    s_dc.job.A      = A + (size_t)M_half * K;  /* 跳过前半行 */
    s_dc.job.B      = B;
    s_dc.job.bias   = bias ? (bias + M_half) : NULL;
    s_dc.job.C      = C + (size_t)M_half * N;  /* 跳过前半行 */
    s_dc.job.M_half = M_rest;
    s_dc.job.N      = N;
    s_dc.job.K      = K;
    s_dc.job.acc    = acc;
    s_dc.job.done   = 0;
    xSemaphoreGive(s_dc.job_sem);

    /* 主核算前半部分 */
    const bnn_op_backend_t *op = BNN_OP();
    op->gemm_nt(A, B, bias, C, M_half, N, K, acc);

    /* 等待 core0 完成 */
    if (!s_dc.job.done) {
        xSemaphoreTake(s_dc.done_sem, portMAX_DELAY);
    }
}

/* 融合版: GEMM + bias + ReLU (用于 BNN_PERF_FUSE_BIAS_RELU) */
void bnn_dual_core_gemm_nt_split_relu(const float *A, const float *B,
                                       const float *bias, float *C,
                                       int M, int N, int K)
{
    /* 先做分割 GEMM (带 bias) */
    bnn_dual_core_gemm_nt_split(A, B, bias, C, M, N, K, 0);
    /* 再统一做 ReLU (很快, 单核即可) */
    size_t total = (size_t)M * N;
    for (size_t i = 0; i < total; ++i) {
        if (C[i] < 0.0f) C[i] = 0.0f;
    }
}

#endif /* BNN_PERF_DUAL_CORE */
