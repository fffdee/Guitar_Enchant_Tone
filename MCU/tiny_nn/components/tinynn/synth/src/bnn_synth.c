#include "bnn_synth/bnn_synth.h"
#include "bnn_synth/bnn_bands.h"
#include "bnn_op/bnn_dsp.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define SYNTH_EPS 1e-8f

#define BNN_SYNTH_MAX_BANDS 32

struct bnn_synth {
    bnn_xform_cfg_t cfg;
    int          band_edges[BNN_SYNTH_MAX_BANDS + 1]; /* bin 索引 (fft_size 频格) */
    int          n_edges;
    unsigned int seed;
};

/* ---------------- 控制率 -> 音频率 线性插值 (与 _upsample_np 一致) ---------------- */
/* 帧序列列向量: 元素 t 位于 v[t*stride]; 返回采样点 i 的插值. np.interp 端点钳位. */
static float interp_at(const float *v, int stride, int T, int hop, int i) {
    double x = (double)i / (double)hop;
    int f = (int)x;
    if (f >= T - 1) return v[(size_t)(T - 1) * stride];
    double frac = x - (double)f;
    float a = v[(size_t)f * stride];
    float b = v[(size_t)(f + 1) * stride];
    return (float)((double)a + ((double)b - (double)a) * frac);
}

/* ---------------- 轻量 RNG: xorshift32 + Box-Muller 高斯 ---------------- */
static unsigned int xs_next(unsigned int *s) {
    unsigned int x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
static float rng_uniform(unsigned int *s) { /* (0,1) */
    unsigned int r = xs_next(s) >> 8;        /* 24 bit */
    return ((float)r + 1.0f) / (float)0x1000001;
}
static void fill_gaussian(float *out, int n, unsigned int seed) {
    unsigned int s = seed ? seed : 0xC2B2AE35u;
    int i = 0;
    for (; i + 1 < n; i += 2) {
        float u1 = rng_uniform(&s), u2 = rng_uniform(&s);
        float r = sqrtf(-2.0f * logf(u1));
        out[i]     = r * cosf(2.0f * (float)M_PI * u2);
        out[i + 1] = r * sinf(2.0f * (float)M_PI * u2);
    }
    if (i < n) {
        float u1 = rng_uniform(&s), u2 = rng_uniform(&s);
        out[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ---------------- 生命周期 ---------------- */
bnn_synth_t *bnn_synth_create(const bnn_xform_cfg_t *cfg) {
    if (!cfg) { BNN_LOGE("synth: null cfg"); return NULL; }
    if (cfg->n_noise_bands > BNN_SYNTH_MAX_BANDS) {
        BNN_LOGE("synth: n_noise_bands %d > %d", cfg->n_noise_bands, BNN_SYNTH_MAX_BANDS);
        return NULL;
    }
    bnn_synth_t *s = (bnn_synth_t *)bnn_calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = *cfg;
    s->seed = 0;
    /* 子带边界: fmin=40 (与 bands.py 默认一致), fmax=Nyquist */
    s->n_edges = bnn_bands_make(cfg->sample_rate, cfg->fft_size, cfg->n_noise_bands,
                                40.0f, -1.0f, s->band_edges);
    if (s->n_edges < 0) { bnn_free(s); return NULL; }
    return s;
}

void bnn_synth_destroy(bnn_synth_t *s) { if (s) bnn_free(s); }

void bnn_synth_set_seed(bnn_synth_t *s, unsigned int seed) { if (s) s->seed = seed; }

const int *bnn_synth_band_edges(const bnn_synth_t *s, int *n_edges) {
    if (!s) return NULL;
    if (n_edges) *n_edges = s->n_edges;
    return s->band_edges;
}

/* ---------------- 谐波加性合成 ---------------- */
int bnn_synth_harmonic(bnn_synth_t *s, const float *f0_hz,
                       const float *harmonic_amp, int T,
                       float *out, int n_samples) {
    if (!s || !f0_hz || !harmonic_amp || !out || T <= 0) return -1;
    const int hop = s->cfg.hop_size;
    const int K   = s->cfg.n_harmonics;
    const double sr = (double)s->cfg.sample_rate;
    const double nyq = sr / 2.0;
    if (n_samples <= 0) n_samples = T * hop;

    double ph = 0.0;  /* 累积相位 (double, 与 numpy cumsum 一致) */
    for (int i = 0; i < n_samples; ++i) {
        /* 帧定位 (一次, k 循环复用) */
        double xpos = (double)i / (double)hop;
        int fr = (int)xpos;
        double frac = 0.0;
        int last = (fr >= T - 1);
        if (!last) frac = xpos - (double)fr;

        double f0u = last ? f0_hz[T - 1]
                          : (double)f0_hz[fr] + ((double)f0_hz[fr + 1] - (double)f0_hz[fr]) * frac;
        ph += 2.0 * M_PI * f0u / sr;

        double acc = 0.0;
        for (int k = 1; k <= K; ++k) {
            if ((double)k * f0u >= nyq) continue;  /* 抗混叠 */
            const float *col = harmonic_amp + (k - 1);
            double amp = last ? col[(size_t)(T - 1) * K]
                              : (double)col[(size_t)fr * K] +
                                ((double)col[(size_t)(fr + 1) * K] - (double)col[(size_t)fr * K]) * frac;
            acc += amp * sin((double)k * ph);
        }
        out[i] = (float)acc;
    }
    return 0;
}

/* ---------------- 子带滤波噪声 (累加到 out) ---------------- */
int bnn_synth_noise(bnn_synth_t *s, const float *noise_band, int T,
                    float *out, int n_samples) {
    if (!s || !noise_band || !out || T <= 0) return -1;
    const int hop = s->cfg.hop_size;
    const int B   = s->cfg.n_noise_bands;
    const double sr = (double)s->cfg.sample_rate;
    const int fft_size = s->cfg.fft_size;
    if (n_samples <= 0) n_samples = T * hop;

    const bnn_dsp_backend_t *dsp = BNN_DSP();
    if (!dsp || !dsp->rfft || !dsp->irfft) { BNN_LOGE("synth: no FFT backend"); return -1; }

    int Nfft = next_pow2(n_samples);
    int nb = Nfft / 2 + 1;

    size_t mark = bnn_ws_mark(NULL);
    float *white = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)Nfft);     /* 复用为 band_sig */
    float *spec  = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)2 * nb);
    float *mspec = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)2 * nb);
    if (!white || !spec || !mspec) { BNN_LOGE("synth noise ws OOM"); bnn_ws_reset_to(NULL, mark); return -1; }

    fill_gaussian(white, Nfft, s->seed);
    dsp->rfft(white, spec, Nfft);          /* spec[2*nb] 交错 */
    float *band_sig = white;               /* 之后复用 white 缓冲 */

    for (int b = 0; b < B; ++b) {
        double hz_lo = (double)s->band_edges[b]     * sr / (double)fft_size;
        double hz_hi = (double)s->band_edges[b + 1] * sr / (double)fft_size;
        /* keep bin j where freq_j=j*sr/Nfft 落在 [hz_lo, hz_hi) */
        int j_lo = (int)ceil(hz_lo * (double)Nfft / sr);
        int j_hi = (int)ceil(hz_hi * (double)Nfft / sr);
        if (j_lo < 0) j_lo = 0;
        if (j_hi > nb) j_hi = nb;
        if (j_hi <= j_lo) continue;        /* 空带 */

        memset(mspec, 0, sizeof(float) * (size_t)2 * nb);
        for (int j = j_lo; j < j_hi; ++j) {
            mspec[2 * j]     = spec[2 * j];
            mspec[2 * j + 1] = spec[2 * j + 1];
        }
        dsp->irfft(mspec, band_sig, Nfft); /* 实数, 已 /Nfft 归一 */

        /* 该带在有效段的 RMS */
        double e = 0.0;
        for (int i = 0; i < n_samples; ++i) e += (double)band_sig[i] * band_sig[i];
        float rms = (float)sqrt(e / (double)n_samples) + SYNTH_EPS;

        const float *gcol = noise_band + b; /* 帧序列, stride=B */
        for (int i = 0; i < n_samples; ++i) {
            float g = interp_at(gcol, B, T, hop, i);
            out[i] += (band_sig[i] / rms) * g;
        }
    }
    bnn_ws_reset_to(NULL, mark);
    return 0;
}

/* ---------------- 完整 DDSP ---------------- */
int bnn_synth_render(bnn_synth_t *s, const float *f0_hz, const float *params,
                     int T, float *out, int n_samples) {
    if (!s || !f0_hz || !params || !out || T <= 0) return -1;
    const int K = s->cfg.n_harmonics;
    const int B = s->cfg.n_noise_bands;
    const int P = K + B;
    const int hop = s->cfg.hop_size;
    if (n_samples <= 0) n_samples = T * hop;

    /* params 逐帧 [T][P]: 前 K 谐波, 后 B 噪声. 直接以子视图喂给两条路径 (stride=P). */
    /* harmonic 期望 [T][K] 连续, noise 期望 [T][B] 连续; 这里 stride 不同, 故拆出连续缓冲. */
    size_t mark = bnn_ws_mark(NULL);
    float *harm  = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)T * K);
    float *noise = (float *)bnn_ws_alloc(NULL, sizeof(float) * (size_t)T * B);
    if (!harm || !noise) { BNN_LOGE("synth render ws OOM"); bnn_ws_reset_to(NULL, mark); return -1; }
    for (int t = 0; t < T; ++t) {
        const float *row = params + (size_t)t * P;
        memcpy(harm  + (size_t)t * K, row,     sizeof(float) * (size_t)K);
        memcpy(noise + (size_t)t * B, row + K, sizeof(float) * (size_t)B);
    }

    int rc = bnn_synth_harmonic(s, f0_hz, harm, T, out, n_samples);
    if (rc == 0) rc = bnn_synth_noise(s, noise, T, out, n_samples);
    bnn_ws_reset_to(NULL, mark);
    return rc;
}
