#include "bnn_frontend/bnn_specfront.h"
#include "bnn_op/bnn_dsp.h"
#include "bnn_op/bnn_op.h"
#include "bnn_specmat.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct bnn_specfront {
    bnn_mask_cfg_t cfg;
    int    n_bins;
    float *window;      /* [n_fft] Hann */
    float *mel_basis;   /* 保留字段(创建后立即释放) */
    float *mel_mean;    /* [n_mels] */
    float *mel_std;     /* [n_mels] */
    float *in_buf;      /* [n_fft] 加窗 */
    float *cplx;        /* [2*n_bins] rfft 输出 */
    float *mag_tmp;     /* [n_bins] */

    /* 稀疏梅尔滤波器: 三角滤波器每列最多 2 个非零 → 消除 193KB PSRAM 读
     * 每 bin 贡献至多 2 个 mel band, 总计 ~6KB → 促入 SRAM */
    uint8_t *mel_sp_i0; /* [n_bins] 第一个 mel 索引 */
    uint8_t *mel_sp_i1; /* [n_bins] 第二个 mel 索引 */
    float   *mel_sp_w0; /* [n_bins] 第一个权重 */
    float   *mel_sp_w1; /* [n_bins] 第二个权重 (=0 若此 bin 只覆盖 1 个 mel band) */
};

bnn_specfront_t *bnn_specfront_create(const bnn_mask_cfg_t *cfg) {
    if (!cfg) { BNN_LOGE("specfront: null cfg"); return NULL; }
    bnn_specfront_t *fe = (bnn_specfront_t *)bnn_calloc(1, sizeof(*fe));
    if (!fe) return NULL;
    fe->cfg = *cfg;
    int n_fft = cfg->n_fft, n_bins = cfg->n_fft / 2 + 1, n_mels = cfg->n_mels;
    fe->n_bins = n_bins;

    fe->window    = (float *)bnn_calloc((size_t)n_fft, sizeof(float));
    fe->mel_basis = (float *)bnn_calloc((size_t)n_mels * n_bins, sizeof(float));
    fe->mel_mean  = (float *)bnn_calloc((size_t)n_mels, sizeof(float));
    fe->mel_std   = (float *)bnn_calloc((size_t)n_mels, sizeof(float));
    fe->in_buf    = (float *)bnn_calloc((size_t)n_fft, sizeof(float));
    fe->cplx      = (float *)bnn_calloc((size_t)2 * n_bins, sizeof(float));
    fe->mag_tmp   = (float *)bnn_calloc((size_t)n_bins, sizeof(float));
    if (!fe->window || !fe->mel_basis || !fe->mel_mean || !fe->mel_std ||
        !fe->in_buf || !fe->cplx || !fe->mag_tmp) { bnn_specfront_destroy(fe); return NULL; }

    for (int k = 0; k < n_fft; ++k)
        fe->window[k] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * (double)k / (double)n_fft));
    if (bnn_specmat_mel_basis(cfg->sample_rate, n_fft, n_mels, cfg->fmin, cfg->fmax, fe->mel_basis) != 0) {
        bnn_specfront_destroy(fe); return NULL;
    }
    for (int m = 0; m < n_mels; ++m) { fe->mel_mean[m] = 0.0f; fe->mel_std[m] = 1.0f; }

    /* === 构建稀疏梅尔滤波器 (按列扫描 mel_basis) ===
     * 三角滤波器: 每个频率 bin 最多贡献 2 个相邻 mel band
     * 释放 193KB 稠密矩阵, 保留 ~6KB 稀疏索引+权重 → 促入 SRAM */
    fe->mel_sp_i0 = (uint8_t *)bnn_calloc((size_t)n_bins, sizeof(uint8_t));
    fe->mel_sp_i1 = (uint8_t *)bnn_calloc((size_t)n_bins, sizeof(uint8_t));
    fe->mel_sp_w0 = (float   *)bnn_calloc((size_t)n_bins, sizeof(float));
    fe->mel_sp_w1 = (float   *)bnn_calloc((size_t)n_bins, sizeof(float));
    if (!fe->mel_sp_i0 || !fe->mel_sp_i1 || !fe->mel_sp_w0 || !fe->mel_sp_w1) {
        bnn_specfront_destroy(fe); return NULL;
    }
    for (int k = 0; k < n_bins; ++k) {
        int m0 = -1, m1 = -1;
        float w0 = 0.0f, w1 = 0.0f;
        for (int m = 0; m < n_mels; ++m) {
            float v = fe->mel_basis[(size_t)m * n_bins + k];
            if (v > 0.0f) {
                if (m0 < 0) { m0 = m; w0 = v; }
                else         { m1 = m; w1 = v; break; }
            }
        }
        fe->mel_sp_i0[k] = (uint8_t)(m0 >= 0 ? m0 : 0);
        fe->mel_sp_i1[k] = (uint8_t)(m1 >= 0 ? m1 : (m0 >= 0 ? m0 : 0));
        fe->mel_sp_w0[k] = w0;
        fe->mel_sp_w1[k] = w1;  /* = 0 时第二次加法贡献为 0, 无需分支 */
    }
    bnn_free(fe->mel_basis);  /* 释放 193KB PSRAM */
    fe->mel_basis = NULL;

    /* 促入内部 SRAM (~6KB 合计) */
    fe->mel_sp_i0 = (uint8_t *)bnn_try_promote_internal(fe->mel_sp_i0, (size_t)n_bins * sizeof(uint8_t));
    fe->mel_sp_i1 = (uint8_t *)bnn_try_promote_internal(fe->mel_sp_i1, (size_t)n_bins * sizeof(uint8_t));
    fe->mel_sp_w0 = (float   *)bnn_try_promote_internal(fe->mel_sp_w0, (size_t)n_bins * sizeof(float));
    fe->mel_sp_w1 = (float   *)bnn_try_promote_internal(fe->mel_sp_w1, (size_t)n_bins * sizeof(float));
    BNN_LOGI("specfront sparse mel: %dKB→6KB, 49248 MAC/帧→1026 MAC/帧",
             (int)((size_t)n_mels * n_bins * 4 / 1024));
    return fe;
}

void bnn_specfront_destroy(bnn_specfront_t *fe) {
    if (!fe) return;
    if (fe->window)    bnn_free(fe->window);
    if (fe->mel_basis) bnn_free(fe->mel_basis);
    if (fe->mel_mean)  bnn_free(fe->mel_mean);
    if (fe->mel_std)   bnn_free(fe->mel_std);
    if (fe->in_buf)    bnn_free(fe->in_buf);
    if (fe->cplx)      bnn_free(fe->cplx);
    if (fe->mag_tmp)   bnn_free(fe->mag_tmp);
    if (fe->mel_sp_i0) bnn_free(fe->mel_sp_i0);
    if (fe->mel_sp_i1) bnn_free(fe->mel_sp_i1);
    if (fe->mel_sp_w0) bnn_free(fe->mel_sp_w0);
    if (fe->mel_sp_w1) bnn_free(fe->mel_sp_w1);
    bnn_free(fe);
}

void bnn_specfront_set_norm(bnn_specfront_t *fe, const float *mel_mean, const float *mel_std) {
    if (!fe) return;
    int n_mels = fe->cfg.n_mels;
    if (mel_mean) memcpy(fe->mel_mean, mel_mean, sizeof(float) * (size_t)n_mels);
    if (mel_std)  memcpy(fe->mel_std,  mel_std,  sizeof(float) * (size_t)n_mels);
}

void bnn_specfront_extract(bnn_specfront_t *fe, const float *frame,
                           float *logmel, float *mag, float *phase) {
    if (!fe || !frame) return;
    const bnn_mask_cfg_t *c = &fe->cfg;
    int n_fft = c->n_fft, n_bins = fe->n_bins, n_mels = c->n_mels;

    const bnn_dsp_backend_t *dsp = BNN_DSP();
    if (!dsp || !dsp->rfft) { BNN_LOGE("specfront: no FFT backend"); return; }

    for (int i = 0; i < n_fft; ++i) fe->in_buf[i] = frame[i] * fe->window[i];
    dsp->rfft(fe->in_buf, fe->cplx, n_fft);

    float *mg = fe->mag_tmp;
    for (int k = 0; k < n_bins; ++k) {
        float re = fe->cplx[2 * k], im = fe->cplx[2 * k + 1];
        float m = sqrtf(re * re + im * im);
        mg[k] = m;
        /* 存单位相位 (cos,sin) 而非 atan2, 避免 513×atan2f/帧 */
        if (phase) {
            float inv = m > 1e-8f ? (1.0f / m) : 0.0f;
            phase[2 * k]     = re * inv;
            phase[2 * k + 1] = im * inv;
        }
    }
    if (mag) memcpy(mag, mg, sizeof(float) * (size_t)n_bins);

    if (logmel) {
        /* 稀疏梅尔滤波: 按 bin 顺序散射累加到 mel band
         * 三角滤波器每列至多 2 个非零, 无分支, 完全 SRAM 访问
         * 消除: 193KB PSRAM mel_basis 矩阵读 + 49248 MAC → 1026 MAC */
        memset(logmel, 0, sizeof(float) * (size_t)n_mels);
        for (int k = 0; k < n_bins; ++k) {
            float v = mg[k];
            logmel[(int)fe->mel_sp_i0[k]] += fe->mel_sp_w0[k] * v;
            logmel[(int)fe->mel_sp_i1[k]] += fe->mel_sp_w1[k] * v;
        }
        for (int m = 0; m < n_mels; ++m) {
            float lm = logf(logmel[m] + c->log_eps);
            /* 与 PC 端 (mel_std + 1e-8) 一致: 极小 std 仍参与除法 */
            float sd = (fe->mel_std[m] > 1e-8f) ? fe->mel_std[m] : 1.0f;
            logmel[m] = (lm - fe->mel_mean[m]) / sd;
        }
    }
}
