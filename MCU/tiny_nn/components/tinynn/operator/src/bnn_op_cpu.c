#include "bnn_op/bnn_op.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* =============== 逐元素 =============== */
static void op_add(const float *a, const float *b, float *c, size_t n) { for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i]; }
static void op_sub(const float *a, const float *b, float *c, size_t n) { for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i]; }
static void op_mul(const float *a, const float *b, float *c, size_t n) { for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i]; }
static void op_div(const float *a, const float *b, float *c, size_t n) { for (size_t i = 0; i < n; ++i) c[i] = a[i] / b[i]; }
static void op_adds(const float *a, float s, float *c, size_t n)        { for (size_t i = 0; i < n; ++i) c[i] = a[i] + s; }
static void op_muls(const float *a, float s, float *c, size_t n)        { for (size_t i = 0; i < n; ++i) c[i] = a[i] * s; }
static void op_axpy(float alpha, const float *x, float *y, size_t n)    { for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i]; }

/* =============== 一元数学 =============== */
static void op_exp(const float *x, float *y, size_t n) { for (size_t i = 0; i < n; ++i) y[i] = expf(x[i]); }
static void op_log(const float *x, float *y, size_t n) { for (size_t i = 0; i < n; ++i) y[i] = logf(x[i]); }
static void op_sqrt(const float *x, float *y, size_t n){ for (size_t i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }

/* =============== GEMM =============== */
static void bias_or_zero(float *C, const float *bias, int M, int N, int acc) {
    if (!acc) {
        if (bias) {
            for (int i = 0; i < M; ++i) memcpy(C + i * N, bias, sizeof(float) * (size_t)N);
        } else {
            memset(C, 0, sizeof(float) * (size_t)M * (size_t)N);
        }
    } else if (bias) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) C[i * N + j] += bias[j];
    }
}

static void op_gemm(const float *A, const float *B, const float *bias,
                    float *C, int M, int N, int K, int acc) {
    bias_or_zero(C, bias, M, N, acc);
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        for (int k = 0; k < K; ++k) {
            float a0 = A[(m + 0) * K + k];
            float a1 = A[(m + 1) * K + k];
            float a2 = A[(m + 2) * K + k];
            float a3 = A[(m + 3) * K + k];
            const float *brow = B + k * N;
            float *c0 = C + (m + 0) * N;
            float *c1 = C + (m + 1) * N;
            float *c2 = C + (m + 2) * N;
            float *c3 = C + (m + 3) * N;
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                float b0 = brow[j], b1 = brow[j+1], b2 = brow[j+2], b3 = brow[j+3];
                c0[j]   += a0 * b0; c0[j+1] += a0 * b1; c0[j+2] += a0 * b2; c0[j+3] += a0 * b3;
                c1[j]   += a1 * b0; c1[j+1] += a1 * b1; c1[j+2] += a1 * b2; c1[j+3] += a1 * b3;
                c2[j]   += a2 * b0; c2[j+1] += a2 * b1; c2[j+2] += a2 * b2; c2[j+3] += a2 * b3;
                c3[j]   += a3 * b0; c3[j+1] += a3 * b1; c3[j+2] += a3 * b2; c3[j+3] += a3 * b3;
            }
            for (; j < N; ++j) {
                float b = brow[j];
                c0[j] += a0 * b; c1[j] += a1 * b; c2[j] += a2 * b; c3[j] += a3 * b;
            }
        }
    }
    for (; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float a = A[m * K + k];
            const float *brow = B + k * N;
            float *crow = C + m * N;
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
}

static void op_gemm_nt(const float *A, const float *B, const float *bias,
                       float *C, int M, int N, int K, int acc) {
    bias_or_zero(C, bias, M, N, acc);
    for (int i = 0; i < M; ++i) {
        const float *ar = A + i * K;
        float *cr = C + i * N;
        for (int j = 0; j < N; ++j) {
            const float *br = B + j * K;
            float s = 0.f;
            int k = 0;
            for (; k + 4 <= K; k += 4)
                s += ar[k]*br[k] + ar[k+1]*br[k+1] + ar[k+2]*br[k+2] + ar[k+3]*br[k+3];
            for (; k < K; ++k) s += ar[k] * br[k];
            cr[j] += s;
        }
    }
}

static void op_gemm_tn(const float *A, const float *B, const float *bias,
                       float *C, int M, int N, int K, int acc) {
    bias_or_zero(C, bias, M, N, acc);
    for (int k = 0; k < K; ++k) {
        const float *arow = A + k * M;
        const float *brow = B + k * N;
        for (int i = 0; i < M; ++i) {
            float a = arow[i];
            float *cr = C + i * N;
            for (int j = 0; j < N; ++j) cr[j] += a * brow[j];
        }
    }
}

static void op_transpose(const float *src, float *dst, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            dst[j * M + i] = src[i * N + j];
}

/* =============== 激活 =============== */
static void op_relu(const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
static void op_relu_grad(const float *x, const float *dy, float *dx, size_t n) {
    for (size_t i = 0; i < n; ++i) dx[i] = x[i] > 0.f ? dy[i] : 0.f;
}
static void op_sigmoid(const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i]));
}
static void op_tanh(const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = tanhf(x[i]);
}
static void op_softmax_rows(const float *x, float *y, int batch, int num_class) {
    for (int b = 0; b < batch; ++b) {
        const float *xr = x + b * num_class;
        float *yr = y + b * num_class;
        float mx = -FLT_MAX;
        for (int i = 0; i < num_class; ++i) if (xr[i] > mx) mx = xr[i];
        float sum = 0.f;
        for (int i = 0; i < num_class; ++i) { yr[i] = expf(xr[i] - mx); sum += yr[i]; }
        float inv = 1.f / sum;
        for (int i = 0; i < num_class; ++i) yr[i] *= inv;
    }
}

/* =============== 归约 =============== */
static float op_sum(const float *x, size_t n) {
    double s = 0; for (size_t i = 0; i < n; ++i) s += x[i]; return (float)s;
}
static float op_dot(const float *a, const float *b, size_t n) {
    double s = 0; for (size_t i = 0; i < n; ++i) s += (double)a[i] * b[i]; return (float)s;
}
static float op_max(const float *x, size_t n) {
    float m = -FLT_MAX; for (size_t i = 0; i < n; ++i) if (x[i] > m) m = x[i]; return m;
}
static int op_argmax(const float *x, size_t n) {
    int idx = 0; float m = -FLT_MAX;
    for (size_t i = 0; i < n; ++i) if (x[i] > m) { m = x[i]; idx = (int)i; }
    return idx;
}

/* =============== im2col / col2im =============== */
static void op_im2col(const float *im, int C, int H, int W,
                      int K, int S, int P, float *col) {
    int Ho = (H + 2 * P - K) / S + 1;
    int Wo = (W + 2 * P - K) / S + 1;
    int cols = Ho * Wo;
    for (int c = 0; c < C; ++c)
    for (int kh = 0; kh < K; ++kh)
    for (int kw = 0; kw < K; ++kw) {
        int row = (c * K + kh) * K + kw;
        float *out = col + (size_t)row * cols;
        for (int oh = 0; oh < Ho; ++oh) {
            int ih = oh * S - P + kh;
            for (int ow = 0; ow < Wo; ++ow) {
                int iw = ow * S - P + kw;
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                    out[oh * Wo + ow] = im[(c * H + ih) * W + iw];
                else
                    out[oh * Wo + ow] = 0.f;
            }
        }
    }
}

static void op_col2im(const float *col, int C, int H, int W,
                      int K, int S, int P, float *im) {
    int Ho = (H + 2 * P - K) / S + 1;
    int Wo = (W + 2 * P - K) / S + 1;
    int cols = Ho * Wo;
    for (int c = 0; c < C; ++c)
    for (int kh = 0; kh < K; ++kh)
    for (int kw = 0; kw < K; ++kw) {
        int row = (c * K + kh) * K + kw;
        const float *in = col + (size_t)row * cols;
        for (int oh = 0; oh < Ho; ++oh) {
            int ih = oh * S - P + kh;
            for (int ow = 0; ow < Wo; ++ow) {
                int iw = ow * S - P + kw;
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                    im[(c * H + ih) * W + iw] += in[oh * Wo + ow];
            }
        }
    }
}

static const bnn_op_backend_t g_cpu = {
    .name = "cpu_ref",
    .add = op_add, .sub = op_sub, .mul = op_mul, .divf = op_div,
    .adds = op_adds, .muls = op_muls, .axpy = op_axpy,
    .expf_v = op_exp, .logf_v = op_log, .sqrtf_v = op_sqrt,
    .gemm = op_gemm, .gemm_nt = op_gemm_nt, .gemm_tn = op_gemm_tn,
    .transpose = op_transpose,
    .relu = op_relu, .relu_grad = op_relu_grad,
    .sigmoid = op_sigmoid, .tanh = op_tanh,
    .softmax_rows = op_softmax_rows,
    .sum = op_sum, .dot = op_dot, .max_v = op_max, .argmax = op_argmax,
    .im2col = op_im2col, .col2im = op_col2im,
};

const bnn_op_backend_t *bnn_op_cpu_backend(void) { return &g_cpu; }
