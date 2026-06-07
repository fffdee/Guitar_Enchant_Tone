#include "bnn_op/bnn_dsp.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * 复数 radix-2 迭代 FFT (Cooley-Tukey), 原地. data 为交错 (re,im), 长度 2*n.
 * dir=-1 正变换 e^{-j2πkn/N} (与 numpy 一致); dir=+1 逆变换 (未归一).
 * twiddle 用 double 累积以抑制 n 较大时的相位误差.
 */
static void cfft(float *data, int n, int dir) {
    if (n < 2) return;
    /* 位反转置换 */
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2 * i],     ti = data[2 * i + 1];
            data[2 * i]     = data[2 * j];     data[2 * i + 1] = data[2 * j + 1];
            data[2 * j]     = tr;              data[2 * j + 1] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / (double)len * (dir < 0 ? -1.0 : 1.0);
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            int half = len >> 1;
            for (int k = 0; k < half; ++k) {
                int i0 = 2 * (i + k);
                int i1 = 2 * (i + k + half);
                float ur = data[i0],     ui = data[i0 + 1];
                float vr = (float)(data[i1] * cwr - data[i1 + 1] * cwi);
                float vi = (float)(data[i1] * cwi + data[i1 + 1] * cwr);
                data[i0]     = ur + vr; data[i0 + 1] = ui + vi;
                data[i1]     = ur - vr; data[i1 + 1] = ui - vi;
                double ncwr = cwr * wr - cwi * wi;
                double ncwi = cwr * wi + cwi * wr;
                cwr = ncwr; cwi = ncwi;
            }
        }
    }
}

static void rfft_cpu(const float *in, float *out_complex, int n) {
    size_t mark = bnn_ws_mark(NULL);
    float *tmp = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)2 * n);
    if (!tmp) { BNN_LOGE("rfft ws OOM"); return; }
    for (int i = 0; i < n; ++i) { tmp[2 * i] = in[i]; tmp[2 * i + 1] = 0.f; }
    cfft(tmp, n, -1);
    int half = n / 2 + 1;
    memcpy(out_complex, tmp, sizeof(float) * (size_t)2 * half);
    bnn_ws_reset_to(NULL, mark);
}

static void irfft_cpu(const float *in_complex, float *out, int n) {
    size_t mark = bnn_ws_mark(NULL);
    float *tmp = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)2 * n);
    if (!tmp) { BNN_LOGE("irfft ws OOM"); return; }
    int half = n / 2 + 1;
    /* 厄米对称重建完整谱 */
    for (int k = 0; k < half; ++k) {
        tmp[2 * k]     = in_complex[2 * k];
        tmp[2 * k + 1] = in_complex[2 * k + 1];
    }
    for (int k = half; k < n; ++k) {
        int src = n - k;                 /* 共轭镜像 */
        tmp[2 * k]     =  in_complex[2 * src];
        tmp[2 * k + 1] = -in_complex[2 * src + 1];
    }
    cfft(tmp, n, +1);
    float inv = 1.0f / (float)n;
    for (int i = 0; i < n; ++i) out[i] = tmp[2 * i] * inv;
    bnn_ws_reset_to(NULL, mark);
}

static const bnn_dsp_backend_t g_cpu = {
    .name = "cpu_radix2",
    .rfft = rfft_cpu,
    .irfft = irfft_cpu,
    .cfft = cfft,
};

static const bnn_dsp_backend_t *g_dsp = &g_cpu;

void                     bnn_dsp_set_backend(const bnn_dsp_backend_t *be) { if (be) g_dsp = be; }
const bnn_dsp_backend_t *bnn_dsp_get_backend(void) { return g_dsp; }
const bnn_dsp_backend_t *bnn_dsp_cpu_backend(void) { return &g_cpu; }
