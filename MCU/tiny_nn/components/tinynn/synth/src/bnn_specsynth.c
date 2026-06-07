#include "bnn_synth/bnn_specsynth.h"
#include "bnn_op/bnn_dsp.h"
#include "bnn_op/bnn_specmat.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define SS_EPS 1e-8f

struct bnn_specsynth {
    bnn_mask_cfg_t cfg;
    int    n_bins;
    float *mel_inv;     /* [n_bins * n_mels] */
    float *phase_inv;   /* [n_bins * phase_bands] */
    float *mel_basis;   /* [noise_bands * n_bins] 用作 noise_fb */
    float *window;      /* [n_fft] */
    float *spec;        /* [2*n_bins] irfft 输入 */
    float *time;        /* [n_fft] irfft 输出 */
    float *ola;         /* [n_fft] 重叠累加 */
    float *wsum;        /* [n_fft] 窗平方累加 */
    float *gain_prev;   /* [n_bins] 增益平滑 */
    int    have_prev;
    float  smooth_a;
    unsigned int rng;
};

static unsigned int xs_next(unsigned int *s) {
    unsigned int x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x; return x;
}
static float rng_angle(unsigned int *s) {            /* [-π, π) */
    unsigned int r = xs_next(s) >> 8;                /* 24-bit */
    float u = (float)r / (float)0x1000000;
    return (2.0f * u - 1.0f) * (float)M_PI;
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
    return s;
}

void bnn_specsynth_destroy(bnn_specsynth_t *s) {
    if (!s) return;
    if (s->mel_inv) bnn_free(s->mel_inv);
    if (s->phase_inv) bnn_free(s->phase_inv);
    if (s->mel_basis) bnn_free(s->mel_basis);
    if (s->window) bnn_free(s->window);
    if (s->spec) bnn_free(s->spec);
    if (s->time) bnn_free(s->time);
    if (s->ola) bnn_free(s->ola);
    if (s->wsum) bnn_free(s->wsum);
    if (s->gain_prev) bnn_free(s->gain_prev);
    bnn_free(s);
}

void bnn_specsynth_reset(bnn_specsynth_t *s) {
    if (!s) return;
    int n_fft = s->cfg.n_fft;
    memset(s->ola, 0, sizeof(float) * (size_t)n_fft);
    memset(s->wsum, 0, sizeof(float) * (size_t)n_fft);
    memset(s->gain_prev, 0, sizeof(float) * (size_t)s->n_bins);
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
    int n_mels = c->n_mels, P = c->phase_bands, B = c->noise_bands;

    const bnn_dsp_backend_t *dsp = BNN_DSP();
    if (!dsp || !dsp->irfft) { BNN_LOGE("specsynth: no FFT backend"); return -1; }

    float a = s->smooth_a;
    for (int r = 0; r < n_bins; ++r) {
        /* 梅尔掩码 -> 线性增益 (+ 帧间平滑) */
        const float *mi = s->mel_inv + (size_t)r * n_mels;
        double g = 0.0;
        for (int m = 0; m < n_mels; ++m) g += (double)mi[m] * mask[m];
        float gain = (float)g;
        if (s->have_prev && a > 0.0f) gain = a * s->gain_prev[r] + (1.0f - a) * gain;
        s->gain_prev[r] = gain;

        /* 相位残差展开 */
        const float *pi = s->phase_inv + (size_t)r * P;
        double dp = 0.0;
        for (int p = 0; p < P; ++p) dp += (double)pi[p] * dphi[p];
        float ph = phase[r] + (float)dp;

        float ylin = gain * mag[r];
        float re = ylin * cosf(ph);
        float im = ylin * sinf(ph);

        if (add_noise && noise) {
            double ns = 0.0;
            for (int b = 0; b < B; ++b) ns += (double)s->mel_basis[(size_t)b * n_bins + r] * noise[b];
            float rp = rng_angle(&s->rng);
            re += (float)ns * cosf(rp);
            im += (float)ns * sinf(rp);
        }
        s->spec[2 * r] = re;
        s->spec[2 * r + 1] = im;
    }
    s->have_prev = 1;

    dsp->irfft(s->spec, s->time, n_fft);   /* 已 /n_fft 归一 */

    /* 加窗 OLA */
    for (int i = 0; i < n_fft; ++i) {
        float w = s->window[i];
        s->ola[i]  += s->time[i] * w;
        s->wsum[i] += w * w;
    }
    /* 输出已完成的 hop 个样点 */
    for (int j = 0; j < hop; ++j) {
        float wn = s->wsum[j] > SS_EPS ? s->wsum[j] : SS_EPS;
        out_hop[j] = s->ola[j] / wn;
    }
    /* 左移 hop */
    memmove(s->ola, s->ola + hop, sizeof(float) * (size_t)(n_fft - hop));
    memset(s->ola + (n_fft - hop), 0, sizeof(float) * (size_t)hop);
    memmove(s->wsum, s->wsum + hop, sizeof(float) * (size_t)(n_fft - hop));
    memset(s->wsum + (n_fft - hop), 0, sizeof(float) * (size_t)hop);
    return 0;
}
