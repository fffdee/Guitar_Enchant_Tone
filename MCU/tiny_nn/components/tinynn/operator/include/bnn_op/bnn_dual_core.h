/* bnn_dual_core.h — 双核并行 GEMM 辅助接口。
 *
 * 仅在 BNN_PERF_DUAL_CORE=1 时有效, 否则这些函数为空实现。
 */
#ifndef BNN_DUAL_CORE_H
#define BNN_DUAL_CORE_H

#include "bnn_utils/bnn_perf.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BNN_PERF_DUAL_CORE

/* 分割 GEMM: 主核算前半 M/2 行, core0 算后半。
 * 参数同 bnn_op_backend_t.gemm_nt。 */
void bnn_dual_core_gemm_nt_split(const float *A, const float *B,
                                  const float *bias, float *C,
                                  int M, int N, int K, int acc);

/* 融合版: GEMM + bias + ReLU */
void bnn_dual_core_gemm_nt_split_relu(const float *A, const float *B,
                                       const float *bias, float *C,
                                       int M, int N, int K);

#else /* BNN_PERF_DUAL_CORE=0: 空实现, 编译时消除 */

static inline void bnn_dual_core_gemm_nt_split(const float *A, const float *B,
                                  const float *bias, float *C,
                                  int M, int N, int K, int acc)
{
    (void)A; (void)B; (void)bias; (void)C;
    (void)M; (void)N; (void)K; (void)acc;
    /* 不应被调用, 编译时 BNN_PERF_DUAL_CORE=0 路径不引用此函数 */
}

static inline void bnn_dual_core_gemm_nt_split_relu(const float *A, const float *B,
                                       const float *bias, float *C,
                                       int M, int N, int K)
{
    (void)A; (void)B; (void)bias; (void)C;
    (void)M; (void)N; (void)K;
}

#endif

#ifdef __cplusplus
}
#endif
#endif /* BNN_DUAL_CORE_H */
