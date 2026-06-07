#ifndef BNN_OP_H
#define BNN_OP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bnn_op_backend {
    const char *name;

    /* 逐元素 */
    void (*add)(const float *a, const float *b, float *c, size_t n);
    void (*sub)(const float *a, const float *b, float *c, size_t n);
    void (*mul)(const float *a, const float *b, float *c, size_t n);
    void (*divf)(const float *a, const float *b, float *c, size_t n);
    void (*adds)(const float *a, float s, float *c, size_t n);
    void (*muls)(const float *a, float s, float *c, size_t n);
    void (*axpy)(float alpha, const float *x, float *y, size_t n);

    /* 一元数学 */
    void (*expf_v)(const float *x, float *y, size_t n);
    void (*logf_v)(const float *x, float *y, size_t n);
    void (*sqrtf_v)(const float *x, float *y, size_t n);

    /*
     * GEMM 系列, C[M,N] = ... + bias_row_broadcast (可 NULL), accumulate=0 写入/1 累加.
     *  gemm:    C = A[M,K]   * B[K,N]
     *  gemm_nt: C = A[M,K]   * B[N,K]^T
     *  gemm_tn: C = A[K,M]^T * B[K,N]
     */
    void (*gemm)(const float *A, const float *B, const float *bias,
                 float *C, int M, int N, int K, int accumulate);
    void (*gemm_nt)(const float *A, const float *B, const float *bias,
                    float *C, int M, int N, int K, int accumulate);
    void (*gemm_tn)(const float *A, const float *B, const float *bias,
                    float *C, int M, int N, int K, int accumulate);

    /* dst[N,M] = src[M,N]^T */
    void (*transpose)(const float *src, float *dst, int M, int N);

    /* 激活 */
    void (*relu)(const float *x, float *y, size_t n);
    void (*relu_grad)(const float *x, const float *dy, float *dx, size_t n);
    void (*sigmoid)(const float *x, float *y, size_t n);
    void (*tanh)(const float *x, float *y, size_t n);

    /* Softmax (数值稳定) */
    void (*softmax_rows)(const float *x, float *y, int batch, int num_class);

    /* 归约 */
    float (*sum)(const float *x, size_t n);
    float (*dot)(const float *a, const float *b, size_t n);
    float (*max_v)(const float *x, size_t n);
    int   (*argmax)(const float *x, size_t n);

    /* im2col / col2im (NCHW 单 batch). col 布局 [C*K*K, Ho*Wo]. */
    void (*im2col)(const float *im, int C, int H, int W,
                   int K, int S, int P, float *col);
    void (*col2im)(const float *col, int C, int H, int W,
                   int K, int S, int P, float *im /* 累加 */);
} bnn_op_backend_t;

void                    bnn_op_set_backend(const bnn_op_backend_t *be);
const bnn_op_backend_t *bnn_op_get_backend(void);
const bnn_op_backend_t *bnn_op_cpu_backend(void);

#define BNN_OP() (bnn_op_get_backend())

#ifdef __cplusplus
}
#endif
#endif
