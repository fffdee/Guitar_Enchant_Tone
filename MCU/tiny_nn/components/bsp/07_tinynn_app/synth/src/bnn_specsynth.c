#include "bnn_synth/bnn_specsynth.h"
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
#define SS_EPS 1e-8f

struct bnn_specsynth {
    bnn_mask_cfg_t cfg;
    int    n_bins;
    float *mel_inv;          /* 保留字段(创建后立即释放, 字段置NULL) */
    float *phase_inv;        /* 保留字段(创建后立即释放, 字段置NULL) */
    float *mel_basis;        /* 保留字段(创建后立即释放, 字段置NULL) */
    float *window;           /* [n_fft] */
    float *spec;             /* [2*n_bins] irfft 输入 */
    float *time;             /* [n_fft] irfft 输出 */
    float *ola;              /* [n_fft] 重叠累加 (环形缓冲) */
    float *wsum;             /* [n_fft] 窗平方累加 */
    int    ola_pos;          /* 环形写指针 [0, n_fft) */
    float *gain_prev;        /* [n_bins] 增益平滑 */
    int    have_prev;
    float  smooth_a;
    unsigned int rng;

    /* 稀疏 mel_inv: mel_inv[n_bins,n_mels] 每行至多 2 个非零 (继承自 mel_basis 三角滤波器)
     * 消除 193KB SRAM 占用, 96-elem dot → 2 乘加, ~5KB 全在 SRAM */
    uint8_t *mel_inv_i0;     /* [n_bins] mel 索引 0 */
    uint8_t *mel_inv_i1;     /* [n_bins] mel 索引 1 */
    float   *mel_inv_w0;     /* [n_bins] 权重 0 */
    float   *mel_inv_w1;     /* [n_bins] 权重 1 (=0 若只有 1 个非零) */

    /* 稀疏 phase_inv: 每行仅 2 个非零 (线性插值), ~5KB → 促入 SRAM */
    uint8_t *phase_sp_i0;    /* [n_bins] */
    uint8_t *phase_sp_i1;    /* [n_bins] */
    float   *phase_sp_w0;    /* [n_bins] */
    float   *phase_sp_w1;    /* [n_bins] */

    /* 转置 noise_fb: [n_bins, noise_bands] 行主序 → 消除列跨步访问 */
    float   *noise_fb_T;     /* [n_bins * noise_bands] */
    float   *noise_shape_buf;/* [n_bins] 每帧噪声增益预算缓冲 */
};

static unsigned int xs_next(unsigned int *s) {
    unsigned int x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x; return x;
}

/* Marsaglia 拒绝采样: 生成单位圆上均匀随机向量, 无需 sincosf
 * 平均 1.27 次迭代; 避免 ~400 cycle 的 sincosf 调用 */
static void rng_unit_vec(unsigned int *s, float *out_c, float *out_s) {
    float u1, u2, r2;
    do {
        u1 = (float)(xs_next(s) >> 8) * (1.0f / (float)0x800000) - 1.0f;
        u2 = (float)(xs_next(s) >> 8) * (1.0f / (float)0x800000) - 1.0f;
        r2 = u1 * u1 + u2 * u2;
    } while (r2 >= 1.0f || r2 == 0.0f);
    float inv = 1.0f / sqrtf(r2);
    *out_c = u1 * inv;
    *out_s = u2 * inv;
}

bnn_specsynth_t *bnn_specsynth_create(const bnn_mask_cfg_t *cfg) {
    if (!cfg) { BNN_LOGE("specsynth: null cfg"); return NULL; }
    bnn_specsynth_t *s = (bnn_specsynth_t *)bnn_calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = *cfg;
    int n_fft = cfg->n_fft, n_bins = cfg->n_fft / 2 + 1;
    s->n_bins = n_bins;
    s->smooth_a = 0.5f;
    s->rng = 0;

    s->mel_inv   = (float *)bnn_calloc((size_t)n_bins * cfg->n_mels, sizeof(float));
    s->phase_inv = (float *)bnn_calloc((size_t)n_bins * cfg->phase_bands, sizeof(float));
    s->mel_basis = (float *)bnn_calloc((size_t)cfg->noise_bands * n_bins, sizeof(float));
    s->window    = (float *)bnn_calloc((size_t)n_fft, sizeof(float));
    s->spec      = (float *)bnn_calloc((size_t)2 * n_bins, sizeof(float));
    s->time      = (float *)bnn_calloc((size_t)n_fft, sizeof(float));
    s->ola       = (float *)bnn_calloc((size_t)n_fft, sizeof(float));
    s->wsum      = (float *)bnn_calloc((size_t)n_fft, sizeof(float));
    s->gain_prev = (float *)bnn_calloc((size_t)n_bins, sizeof(float));
    if (!s->mel_inv || !s->phase_inv || !s->mel_basis || !s->window || !s->spec ||
        !s->time || !s->ola || !s->wsum || !s->gain_prev) { bnn_specsynth_destroy(s); return NULL; }

    /* 构建展开矩阵 (与 PC 端一致): mel_basis -> mel_inv; noise_fb=粗梅尔; phase_inv 线性插值 */
    float *mb = (float *)bnn_calloc((size_t)cfg->n_mels * n_bins, sizeof(float));
    if (!mb) { bnn_specsynth_destroy(s); return NULL; }
    if (bnn_specmat_mel_basis(cfg->sample_rate, n_fft, cfg->n_mels, cfg->fmin, cfg->fmax, mb) != 0 ||
        bnn_specmat_mel_inv(mb, cfg->n_mels, n_bins, s->mel_inv) != 0 ||
        bnn_specmat_phase_inv(n_bins, cfg->phase_bands, s->phase_inv) != 0 ||
        bnn_specmat_mel_basis(cfg->sample_rate, n_fft, cfg->noise_bands, cfg->fmin, cfg->fmax, s->mel_basis) != 0) {
        bnn_free(mb); bnn_specsynth_destroy(s); return NULL;
    }
    bnn_free(mb);

    for (int k = 0; k < n_fft; ++k)
        s->window[k] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * (double)k / (double)n_fft));

    /* === 稀疏 mel_inv: 释放 193KB, 替换为 ~5KB 稀疏结构 ===
     * mel_inv[r,:] = mel_basis[:,r] 归一化, 继承三角滤波器稀疏性
     * 每行至多 2 个非零 → 96-elem dot product → 2 乘加 */
    {
        int n_mels_loc = cfg->n_mels;
        s->mel_inv_i0 = (uint8_t *)bnn_calloc((size_t)n_bins, sizeof(uint8_t));
        s->mel_inv_i1 = (uint8_t *)bnn_calloc((size_t)n_bins, sizeof(uint8_t));
        s->mel_inv_w0 = (float   *)bnn_calloc((size_t)n_bins, sizeof(float));
        s->mel_inv_w1 = (float   *)bnn_calloc((size_t)n_bins, sizeof(float));
        if (!s->mel_inv_i0 || !s->mel_inv_i1 || !s->mel_inv_w0 || !s->mel_inv_w1) {
            bnn_specsynth_destroy(s); return NULL;
        }
        for (int r = 0; r < n_bins; ++r) {
            const float *row = s->mel_inv + (size_t)r * n_mels_loc;
            int m0 = -1, m1 = -1;
            float w0 = 0.0f, w1 = 0.0f;
            for (int m = 0; m < n_mels_loc; ++m) {
                if (row[m] > 0.0f) {
                    if (m0 < 0) { m0 = m; w0 = row[m]; }
                    else         { m1 = m; w1 = row[m]; break; }
                }
            }
            s->mel_inv_i0[r] = (uint8_t)(m0 >= 0 ? m0 : 0);
            s->mel_inv_i1[r] = (uint8_t)(m1 >= 0 ? m1 : (m0 >= 0 ? m0 : 0));
            s->mel_inv_w0[r] = w0;
            s->mel_inv_w1[r] = w1;
        }
        bnn_free(s->mel_inv);  /* 释放 193KB (PSRAM 或 SRAM) */
        s->mel_inv = NULL;
        s->mel_inv_i0 = (uint8_t *)bnn_try_promote_internal(s->mel_inv_i0, (size_t)n_bins * sizeof(uint8_t));
        s->mel_inv_i1 = (uint8_t *)bnn_try_promote_internal(s->mel_inv_i1, (size_t)n_bins * sizeof(uint8_t));
        s->mel_inv_w0 = (float   *)bnn_try_promote_internal(s->mel_inv_w0, (size_t)n_bins * sizeof(float));
        s->mel_inv_w1 = (float   *)bnn_try_promote_internal(s->mel_inv_w1, (size_t)n_bins * sizeof(float));
    }

    /* === 稀疏 phase_inv (释放 128KB 稠密矩阵, 仅保留 ~5KB) === */
    s->phase_sp_i0 = (uint8_t *)bnn_calloc((size_t)n_bins, sizeof(uint8_t));
    s->phase_sp_i1 = (uint8_t *)bnn_calloc((size_t)n_bins, sizeof(uint8_t));
    s->phase_sp_w0 = (float   *)bnn_calloc((size_t)n_bins, sizeof(float));
    s->phase_sp_w1 = (float   *)bnn_calloc((size_t)n_bins, sizeof(float));
    if (!s->phase_sp_i0 || !s->phase_sp_i1 || !s->phase_sp_w0 || !s->phase_sp_w1) {
        bnn_specsynth_destroy(s); return NULL;
    }
    for (int r = 0; r < n_bins; ++r) {
        const float *pi = s->phase_inv + (size_t)r * cfg->phase_bands;
        int j0 = -1, j1 = -1;
        float w0 = 0.0f, w1 = 0.0f;
        for (int p = 0; p < cfg->phase_bands; ++p) {
            if (pi[p] != 0.0f) {
                if (j0 < 0) { j0 = p; w0 = pi[p]; }
                else         { j1 = p; w1 = pi[p]; break; }
            }
        }
        s->phase_sp_i0[r] = (uint8_t)(j0 >= 0 ? j0 : 0);
        s->phase_sp_i1[r] = (uint8_t)(j1 >= 0 ? j1 : (j0 >= 0 ? j0 : 0));
        s->phase_sp_w0[r] = w0;
        s->phase_sp_w1[r] = w1;
    }
    bnn_free(s->phase_inv);  /* 释放 128KB 稠密矩阵 (PSRAM) */
    s->phase_inv = NULL;

    /* 小型热数组 → 促入内部 SRAM */
    s->phase_sp_i0 = (uint8_t *)bnn_try_promote_internal(s->phase_sp_i0, (size_t)n_bins * sizeof(uint8_t));
    s->phase_sp_i1 = (uint8_t *)bnn_try_promote_internal(s->phase_sp_i1, (size_t)n_bins * sizeof(uint8_t));
    s->phase_sp_w0 = (float   *)bnn_try_promote_internal(s->phase_sp_w0, (size_t)n_bins * sizeof(float));
    s->phase_sp_w1 = (float   *)bnn_try_promote_internal(s->phase_sp_w1, (size_t)n_bins * sizeof(float));

    /* === 转置 noise_fb: [noise_bands, n_bins] → [n_bins, noise_bands] === */
    int B = cfg->noise_bands;
    s->noise_fb_T = (float *)bnn_calloc((size_t)n_bins * B, sizeof(float));
    if (!s->noise_fb_T) { bnn_specsynth_destroy(s); return NULL; }
    for (int r = 0; r < n_bins; ++r)
        for (int b = 0; b < B; ++b)
            s->noise_fb_T[(size_t)r * B + b] = s->mel_basis[(size_t)b * n_bins + r];
    bnn_free(s->mel_basis);  /* 释放 ~33KB PSRAM */
    s->mel_basis = NULL;
    /* noise_fb_T ~33KB: 尽量促入 SRAM, 失败则保留 PSRAM (行主序访问仍高效) */
    s->noise_fb_T = (float *)bnn_try_promote_internal(s->noise_fb_T, (size_t)n_bins * B * sizeof(float));

    /* 噪声增益预算缓冲 (2KB) → 强烈希望在 SRAM */
    s->noise_shape_buf = (float *)bnn_calloc((size_t)n_bins, sizeof(float));
    if (!s->noise_shape_buf) { bnn_specsynth_destroy(s); return NULL; }
    s->noise_shape_buf = (float *)bnn_try_promote_internal(s->noise_shape_buf, (size_t)n_bins * sizeof(float));

    BNN_LOGI("specsynth sparse: mel_inv %dKB→5KB, phase_inv %dKB→5KB, noise_fb_T=%dKB (seq), Marsaglia noise",
             (int)((size_t)n_bins * cfg->n_mels * 4 / 1024),
             (int)((size_t)n_bins * cfg->phase_bands * 4 / 1024),
             (int)((size_t)n_bins * B * 4 / 1024));
    return s;
}

void bnn_specsynth_destroy(bnn_specsynth_t *s) {
    if (!s) return;
    if (s->mel_inv)          bnn_free(s->mel_inv);    /* 通常为 NULL (已稀疏化) */
    if (s->phase_inv)        bnn_free(s->phase_inv);  /* 通常为 NULL (已稀疏化) */
    if (s->mel_basis)        bnn_free(s->mel_basis);  /* 通常为 NULL (已稀疏化) */
    if (s->mel_inv_i0)       bnn_free(s->mel_inv_i0);
    if (s->mel_inv_i1)       bnn_free(s->mel_inv_i1);
    if (s->mel_inv_w0)       bnn_free(s->mel_inv_w0);
    if (s->mel_inv_w1)       bnn_free(s->mel_inv_w1);
    if (s->window)           bnn_free(s->window);
    if (s->spec)             bnn_free(s->spec);
    if (s->time)             bnn_free(s->time);
    if (s->ola)              bnn_free(s->ola);
    if (s->wsum)             bnn_free(s->wsum);
    if (s->gain_prev)        bnn_free(s->gain_prev);
    if (s->phase_sp_i0)      bnn_free(s->phase_sp_i0);
    if (s->phase_sp_i1)      bnn_free(s->phase_sp_i1);
    if (s->phase_sp_w0)      bnn_free(s->phase_sp_w0);
    if (s->phase_sp_w1)      bnn_free(s->phase_sp_w1);
    if (s->noise_fb_T)       bnn_free(s->noise_fb_T);
    if (s->noise_shape_buf)  bnn_free(s->noise_shape_buf);
    bnn_free(s);
}

void bnn_specsynth_reset(bnn_specsynth_t *s) {
    if (!s) return;
    int n_fft = s->cfg.n_fft;
    memset(s->ola, 0, sizeof(float) * (size_t)n_fft);
    memset(s->wsum, 0, sizeof(float) * (size_t)n_fft);
    memset(s->gain_prev, 0, sizeof(float) * (size_t)s->n_bins);
    s->ola_pos = 0;
    s->have_prev = 0;
}

void bnn_specsynth_set_seed(bnn_specsynth_t *s, unsigned int seed) { if (s) s->rng = seed; }
void bnn_specsynth_set_smooth(bnn_specsynth_t *s, float a) { if (s) s->smooth_a = a; }

int bnn_specsynth_process(bnn_specsynth_t *s, const float *mag, const float *phase,
                          const float *mask, const float *dphi, const float *noise,
                          int add_noise, float *out_hop) {
    if (!s || !mag || !phase || !mask || !dphi || !out_hop) return -1;
    const bnn_mask_cfg_t *c = &s->cfg;
    int n_fft = c->n_fft, n_bins = s->n_bins, hop = c->hop;
    int B = c->noise_bands;
    (void)c->n_mels;      /* 已稀疏化, process 中不再需要 n_mels */
    (void)c->phase_bands; /* 已替换为稀疏索引, 不再需要 P */

    const bnn_dsp_backend_t *dsp = BNN_DSP();
    if (!dsp || !dsp->irfft) { BNN_LOGE("specsynth: no FFT backend"); return -1; }

    /* === 噪声增益预算: 提前做一次行主序 GEMV, 消除循环内的列跨步访问 ===
     * noise_fb_T[n_bins, B]: 每行顺序读 B 个 float (64B), 合计 32KB 顺序读/帧
     * 原始方案: mel_basis[b * n_bins + r] 步长 2052B → 8208 次 PSRAM 随机读/帧 */
    float *noise_shape = NULL;
    if (add_noise && noise && s->noise_fb_T && s->noise_shape_buf) {
        noise_shape = s->noise_shape_buf;
        for (int r = 0; r < n_bins; ++r) {
            float ns = 0.0f;
            const float *fb = s->noise_fb_T + (size_t)r * B;
            for (int b = 0; b < B; ++b) ns += fb[b] * noise[b];
            noise_shape[r] = ns;
        }
    }

    float a = s->smooth_a;
    for (int r = 0; r < n_bins; ++r) {
        /* 稀疏 mel_inv: 2 乘加替代 96-elem dot product (消除 193KB SRAM 矩阵访问)
         * mel_inv 每行至多 2 个非零, 继承自三角滤波器稀疏性 */
        float g = s->mel_inv_w0[r] * mask[(int)s->mel_inv_i0[r]]
                + s->mel_inv_w1[r] * mask[(int)s->mel_inv_i1[r]];
        float gain = g;
        if (s->have_prev && a > 0.0f) gain = a * s->gain_prev[r] + (1.0f - a) * gain;
        s->gain_prev[r] = gain;

        /* 稀疏相位残差: 每行仅 2 个非零, 2次乘加替代 64 次条件跳转
         * 释放了 128KB PSRAM phase_inv 稠密矩阵 */
        float dp = s->phase_sp_w0[r] * dphi[(int)s->phase_sp_i0[r]]
                 + s->phase_sp_w1[r] * dphi[(int)s->phase_sp_i1[r]];

        float ylin = gain * mag[r];
        float pc = phase[2 * r], ps = phase[2 * r + 1];
        float cd, sd;
        sincosf(dp, &sd, &cd);
        float re = ylin * (pc * cd - ps * sd);
        float im = ylin * (pc * sd + ps * cd);

        if (noise_shape) {
            /* Marsaglia 随机单位向量: 替代 rng_angle + sincosf (~400 cycle)
             * 仅用 2×XOR-shift + sqrtf (~30 cycle) */
            float nr, ni;
            rng_unit_vec(&s->rng, &nr, &ni);
            re += noise_shape[r] * nr;
            im += noise_shape[r] * ni;
        }
        s->spec[2 * r] = re;
        s->spec[2 * r + 1] = im;
    }
    s->have_prev = 1;

    dsp->irfft(s->spec, s->time, n_fft);   /* 已 /n_fft 归一 */

    /* 加窗 OLA (环形缓冲, 避免每帧 memmove n_fft 样点) */
    int pos = s->ola_pos;
    for (int i = 0; i < n_fft; ++i) {
        int idx = pos + i;
        if (idx >= n_fft) idx -= n_fft;
        float w = s->window[i];
        s->ola[idx]  += s->time[i] * w;
        s->wsum[idx] += w * w;
    }
    for (int j = 0; j < hop; ++j) {
        int idx = pos + j;
        if (idx >= n_fft) idx -= n_fft;
        float wn = s->wsum[idx] > SS_EPS ? s->wsum[idx] : SS_EPS;
        out_hop[j] = s->ola[idx] / wn;
        s->ola[idx]  = 0.0f;
        s->wsum[idx] = 0.0f;
    }
    s->ola_pos = pos + hop;
    if (s->ola_pos >= n_fft) s->ola_pos -= n_fft;
    return 0;
}

void bnn_specsynth_peek_gain_lin(const bnn_specsynth_t *s, const float *mask,
                                 float *gain_lin, int n) {
    if (!s || !mask || !gain_lin || n <= 0) return;
    int nb = s->n_bins;
    if (n > nb) n = nb;
    for (int r = 0; r < n; ++r) {
        gain_lin[r] = s->mel_inv_w0[r] * mask[(int)s->mel_inv_i0[r]]
                    + s->mel_inv_w1[r] * mask[(int)s->mel_inv_i1[r]];
    }
}
