#include "bnn_frontend/bnn_frontend.h"
#include "bnn_op/bnn_dsp.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define FE_EPS 1e-10f

/*
 * 吉他特征前端 (C 实现, 与 src/esp_xform/features 数值对齐).
 * 仅依赖 op 层的 FFT 后端 (bnn_dsp), 其余纯 C.
 */
struct bnn_frontend {
    bnn_xform_cfg_t cfg;
    int    n_bins;
    int    tau_min, tau_max, yin_w;
    float *window;      /* [frame_size] Hann */
    float *mel_fb;      /* [n_mels * n_bins] */
    float *in_buf;      /* [fft_size] 加窗补零 */
    float *cplx;        /* [2*n_bins] */
    float *power;       /* [n_bins] */
    float *mag;         /* [n_bins] */
    float *mel_energy;  /* [n_mels] */
    float *prev_mag;    /* [n_bins] 上一帧, 算 flux */
    int    have_prev;
    /* YIN 暂存 */
    double *cumsum_e;   /* [frame_size+1] */
    float  *dprime;     /* [tau_max+1] */
};

/* ---------------- Mel / DCT ---------------- */
static double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

static void build_mel_fb(bnn_frontend_t *fe) {
    const bnn_xform_cfg_t *c = &fe->cfg;
    int n_bins = fe->n_bins, n_mels = c->n_mels;
    double mmin = hz_to_mel(c->fmin), mmax = hz_to_mel(c->fmax);
    for (int i = 0; i < n_mels * n_bins; ++i) fe->mel_fb[i] = 0.f;
    for (int m = 1; m <= n_mels; ++m) {
        double fl = mel_to_hz(mmin + (mmax - mmin) * (m - 1) / (n_mels + 1));
        double fc = mel_to_hz(mmin + (mmax - mmin) * (m    ) / (n_mels + 1));
        double fr = mel_to_hz(mmin + (mmax - mmin) * (m + 1) / (n_mels + 1));
        if (fr <= fl) continue;
        float *row = fe->mel_fb + (size_t)(m - 1) * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            double f = (double)k * c->sample_rate / c->fft_size;  /* bin 频率 */
            double left  = (f - fl) / (fc - fl > 1e-9 ? fc - fl : 1e-9);
            double right = (fr - f) / (fr - fc > 1e-9 ? fr - fc : 1e-9);
            double v = left < right ? left : right;
            row[k] = (float)(v > 0.0 ? v : 0.0);
        }
    }
}

/* 正交归一化 DCT-II: 取前 n_out 个系数 (与 Python dct_ii 一致) */
static void dct_ii(const float *x, int n_in, float *out, int n_out) {
    double s0 = sqrt(1.0 / n_in), sk = sqrt(2.0 / n_in);
    for (int k = 0; k < n_out; ++k) {
        double acc = 0.0;
        for (int n = 0; n < n_in; ++n)
            acc += x[n] * cos(M_PI * (2.0 * n + 1.0) * k / (2.0 * n_in));
        out[k] = (float)(acc * (k == 0 ? s0 : sk));
    }
}

/* ---------------- YIN ---------------- */
/* 返回 voiced; 写 *f0_out. 与 f0_yin.py 一致. */
static float yin_f0(bnn_frontend_t *fe, const float *frame, float *f0_out) {
    const bnn_xform_cfg_t *c = &fe->cfg;
    int n = c->frame_size;
    double e = 0.0;
    for (int i = 0; i < n; ++i) e += (double)frame[i] * frame[i];
    double rms = sqrt(e / n);
    if (rms < 1e-4) { *f0_out = 0.f; return 0.f; }

    int tau_min = fe->tau_min, tau_max = fe->tau_max, w = fe->yin_w;
    if (w <= 1) { *f0_out = 0.f; return 0.f; }

    double *cum = fe->cumsum_e;       /* cum[i] = Σ_{j<i} x[j]^2 */
    cum[0] = 0.0;
    for (int i = 0; i < n; ++i) cum[i + 1] = cum[i] + (double)frame[i] * frame[i];
    double term1 = cum[w] - cum[0];

    float *dp = fe->dprime;
    dp[0] = 1.f;
    double running = 0.0;
    for (int tau = 1; tau <= tau_max; ++tau) {
        double cross = 0.0;
        for (int j = 0; j < w; ++j) cross += (double)frame[j] * frame[j + tau];
        double term2 = cum[tau + w] - cum[tau];
        double d = term1 + term2 - 2.0 * cross;
        if (d < 0) d = 0;
        running += d;
        dp[tau] = (float)(running > 1e-12 ? d * tau / running : 1.0);
    }

    int tau = -1;
    for (int t = tau_min; t <= tau_max; ++t) {
        if (dp[t] < 0.1f) {
            while (t + 1 <= tau_max && dp[t + 1] < dp[t]) ++t;
            tau = t; break;
        }
    }
    if (tau < 0) {
        int best = tau_min; float bv = dp[tau_min];
        for (int t = tau_min + 1; t <= tau_max; ++t) if (dp[t] < bv) { bv = dp[t]; best = t; }
        tau = best;
    }
    float aper = dp[tau];
    /* 抛物线插值 */
    double tau_ref = tau;
    if (tau > 0 && tau < tau_max) {
        double a = dp[tau - 1], b = dp[tau], cc = dp[tau + 1];
        double denom = a - 2.0 * b + cc;
        if (fabs(denom) > 1e-12) tau_ref = tau + 0.5 * (a - cc) / denom;
    }
    if (tau_ref <= 0) { *f0_out = 0.f; return 0.f; }
    double f0 = (double)c->sample_rate / tau_ref;
    float voiced = (aper < 0.3f && f0 >= c->f0_min && f0 <= c->f0_max) ? 1.f : 0.f;
    if (f0 < c->f0_min) f0 = c->f0_min;
    if (f0 > c->f0_max) f0 = c->f0_max;
    *f0_out = (float)f0;
    return voiced;
}

/* ---------------- 生命周期 ---------------- */
bnn_frontend_t *bnn_frontend_create(const bnn_xform_cfg_t *cfg) {
    if (!cfg) return NULL;
    bnn_frontend_t *fe = (bnn_frontend_t *)bnn_calloc(1, sizeof(*fe));
    if (!fe) return NULL;
    fe->cfg = *cfg;
    fe->n_bins = cfg->fft_size / 2 + 1;
    fe->tau_min = (int)floor((double)cfg->sample_rate / cfg->f0_max); if (fe->tau_min < 2) fe->tau_min = 2;
    fe->tau_max = (int)ceil((double)cfg->sample_rate / cfg->f0_min);
    fe->yin_w = cfg->frame_size / 2;
    if (fe->yin_w + fe->tau_max >= cfg->frame_size) fe->yin_w = cfg->frame_size - fe->tau_max - 1;

    fe->window     = (float *)bnn_calloc(cfg->frame_size, sizeof(float));
    fe->mel_fb     = (float *)bnn_calloc((size_t)cfg->n_mels * fe->n_bins, sizeof(float));
    fe->in_buf     = (float *)bnn_calloc(cfg->fft_size, sizeof(float));
    fe->cplx       = (float *)bnn_calloc((size_t)2 * fe->n_bins, sizeof(float));
    fe->power      = (float *)bnn_calloc(fe->n_bins, sizeof(float));
    fe->mag        = (float *)bnn_calloc(fe->n_bins, sizeof(float));
    fe->mel_energy = (float *)bnn_calloc(cfg->n_mels, sizeof(float));
    fe->prev_mag   = (float *)bnn_calloc(fe->n_bins, sizeof(float));
    fe->cumsum_e   = (double *)bnn_calloc((size_t)cfg->frame_size + 1, sizeof(double));
    fe->dprime     = (float *)bnn_calloc((size_t)fe->tau_max + 1, sizeof(float));
    if (!fe->window || !fe->mel_fb || !fe->in_buf || !fe->cplx || !fe->power ||
        !fe->mag || !fe->mel_energy || !fe->prev_mag || !fe->cumsum_e || !fe->dprime) {
        bnn_frontend_destroy(fe); return NULL;
    }
    for (int i = 0; i < cfg->frame_size; ++i)
        fe->window[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / cfg->frame_size));
    build_mel_fb(fe);
    fe->have_prev = 0;
    return fe;
}

void bnn_frontend_destroy(bnn_frontend_t *fe) {
    if (!fe) return;
    bnn_free(fe->window); bnn_free(fe->mel_fb); bnn_free(fe->in_buf); bnn_free(fe->cplx);
    bnn_free(fe->power); bnn_free(fe->mag); bnn_free(fe->mel_energy); bnn_free(fe->prev_mag);
    bnn_free(fe->cumsum_e); bnn_free(fe->dprime);
    bnn_free(fe);
}

void bnn_frontend_reset(bnn_frontend_t *fe) { if (fe) fe->have_prev = 0; }

/* ---------------- 单帧特征 ---------------- */
void bnn_frontend_extract(bnn_frontend_t *fe, const float *frame,
                          float *feat, float *f0_hz, float *voiced) {
    const bnn_xform_cfg_t *c = &fe->cfg;
    int n = c->frame_size, n_bins = fe->n_bins, nyq_idx = c->fft_size / 2;
    float nyquist = c->sample_rate * 0.5f;

    /* 响度 (原始帧 RMS) */
    double e = 0.0;
    for (int i = 0; i < n; ++i) e += (double)frame[i] * frame[i];
    float rms = (float)sqrt(e / n + 1e-12);
    float loud_db = 20.f * log10f(rms > FE_EPS ? rms : FE_EPS);
    int silence = loud_db < c->silence_db;

    /* 加窗 + 补零 + rfft */
    for (int i = 0; i < n; ++i) fe->in_buf[i] = frame[i] * fe->window[i];
    for (int i = n; i < c->fft_size; ++i) fe->in_buf[i] = 0.f;
    BNN_DSP()->rfft(fe->in_buf, fe->cplx, c->fft_size);
    double mag_sum = 0.0, pow_total = 0.0, centroid_acc = 0.0;
    for (int k = 0; k < n_bins; ++k) {
        float re = fe->cplx[2 * k], im = fe->cplx[2 * k + 1];
        float p = re * re + im * im;
        float m = sqrtf(p);
        fe->power[k] = p; fe->mag[k] = m;
        float freq = (float)k * c->sample_rate / c->fft_size;
        mag_sum += m; pow_total += p; centroid_acc += (double)m * freq;
    }

    /* MFCC */
    for (int m = 0; m < c->n_mels; ++m) {
        const float *row = fe->mel_fb + (size_t)m * n_bins;
        double acc = 0.0;
        for (int k = 0; k < n_bins; ++k) acc += (double)row[k] * fe->power[k];
        fe->mel_energy[m] = (float)log(acc > FE_EPS ? acc : FE_EPS);
    }
    float mfcc[64];
    dct_ii(fe->mel_energy, c->n_mels, mfcc, c->n_mfcc);

    /* 质心 / 滚降 (归一化) */
    float centroid_norm = (float)(centroid_acc / (mag_sum + FE_EPS)) / nyquist;
    double thresh = c->rolloff_percent * (pow_total + FE_EPS);
    double cum = 0.0; int roll_idx = 0;
    for (int k = 0; k < n_bins; ++k) { cum += fe->power[k]; if (cum >= thresh) { roll_idx = k; break; } }
    float rolloff_norm = ((float)roll_idx * c->sample_rate / c->fft_size) / nyquist;
    (void)nyq_idx;

    /* 过零率 (原始帧) */
    int zc = 0; float prev = frame[0] >= 0 ? 1.f : -1.f;
    for (int i = 1; i < n; ++i) { float s = frame[i] >= 0 ? 1.f : -1.f; if (s != prev) zc++; prev = s; }
    float zcr = (float)zc / (float)n;

    /* 频谱通量 */
    float flux = 0.f;
    if (fe->have_prev) {
        double acc = 0.0;
        for (int k = 0; k < n_bins; ++k) { float d = fe->mag[k] - fe->prev_mag[k]; if (d > 0) acc += (double)d * d; }
        flux = (float)(sqrt(acc) / sqrt((double)n_bins));
    }
    memcpy(fe->prev_mag, fe->mag, sizeof(float) * n_bins);
    fe->have_prev = 1;

    /* 基频 */
    float f0 = 0.f, vflag = 0.f;
    vflag = yin_f0(fe, frame, &f0);
    if (silence) { vflag = 0.f; f0 = 0.f; }
    float f0_safe = f0 > 0.f ? f0 : c->f0_min;
    if (f0_safe < c->f0_min) f0_safe = c->f0_min;
    if (f0_safe > c->f0_max) f0_safe = c->f0_max;
    float log_f0 = logf(f0_safe);

    /* 组装 20 维 */
    int idx = 0;
    feat[idx++] = log_f0;
    feat[idx++] = loud_db;
    for (int i = 0; i < c->n_mfcc; ++i) feat[idx++] = mfcc[i];
    feat[idx++] = centroid_norm;
    feat[idx++] = rolloff_norm;
    feat[idx++] = vflag;
    feat[idx++] = zcr;
    feat[idx++] = flux;
    /* 维度不足/超出的保护 */
    for (; idx < c->feature_dim; ++idx) feat[idx] = 0.f;

    if (f0_hz)  *f0_hz  = f0;
    if (voiced) *voiced = vflag;
}

void bnn_frontend_standardize(const bnn_xform_cfg_t *cfg, const float *mean,
                              const float *std, float *feat) {
    for (int i = 0; i < cfg->feature_dim; ++i)
        feat[i] = (feat[i] - mean[i]) / (std[i] + 1e-7f);
}
