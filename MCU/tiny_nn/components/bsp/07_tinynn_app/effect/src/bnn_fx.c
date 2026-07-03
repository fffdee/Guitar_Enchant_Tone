#include "bnn_fx/bnn_fx.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 线性 FIR + 循环延迟线 (流式) */
typedef struct {
    int    taps;
    float *h;     /* [taps] */
    float *z;     /* [taps] 延迟线 */
    int    pos;
} fir_t;

struct bnn_fx {
    bnn_fx_cfg_t cfg;
    int   L;
    fir_t up;     /* 插值低通 (含增益 L) */
    fir_t down;   /* 抗混叠低通 */
    int   phase;  /* 抽取相位计数 */
};

static int fir_init(fir_t *f, int taps, double fc, double gain) {
    f->taps = taps;
    f->h = (float *)bnn_calloc((size_t)taps, sizeof(float));
    f->z = (float *)bnn_calloc((size_t)taps, sizeof(float));
    if (!f->h || !f->z) return -1;
    double c = (taps - 1) / 2.0, sum = 0.0;
    for (int i = 0; i < taps; ++i) {
        double t = 2.0 * fc * ((double)i - c);
        double sinc = (fabs(t) < 1e-9) ? 1.0 : sin(M_PI * t) / (M_PI * t);
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / (double)(taps - 1)); /* Hamming */
        f->h[i] = (float)(sinc * w);
        sum += f->h[i];
    }
    for (int i = 0; i < taps; ++i) f->h[i] = (float)(f->h[i] / sum * gain);
    f->pos = 0;
    return 0;
}
static void fir_free(fir_t *f) { if (f->h) bnn_free(f->h); if (f->z) bnn_free(f->z); f->h = f->z = NULL; }
static void fir_reset(fir_t *f) { if (f->z) memset(f->z, 0, sizeof(float) * (size_t)f->taps); f->pos = 0; }
static float fir_step(fir_t *f, float x) {
    f->z[f->pos] = x;
    double acc = 0.0;
    int idx = f->pos;
    for (int k = 0; k < f->taps; ++k) {
        acc += (double)f->h[k] * f->z[idx];
        idx = idx ? idx - 1 : f->taps - 1;
    }
    f->pos = (f->pos + 1) % f->taps;
    return (float)acc;
}

static float shape(const bnn_fx_t *fx, float x) {
    float u = fx->cfg.drive * x;
    switch (fx->cfg.shaper) {
        case BNN_FX_CUBIC: {
            float uc = u < -1.f ? -1.f : (u > 1.f ? 1.f : u);
            return (uc - uc * uc * uc / 3.0f) * 1.5f;
        }
        case BNN_FX_HARD:
            return u < -1.f ? -1.f : (u > 1.f ? 1.f : u);
        case BNN_FX_TANH:
        default:
            return tanhf(u);
    }
}

bnn_fx_t *bnn_fx_create(const bnn_fx_cfg_t *cfg) {
    if (!cfg) { BNN_LOGE("fx: null cfg"); return NULL; }
    bnn_fx_t *fx = (bnn_fx_t *)bnn_calloc(1, sizeof(*fx));
    if (!fx) return NULL;
    fx->cfg = *cfg;
    int L = cfg->oversample;
    if (L != 1 && L != 2 && L != 4) { BNN_LOGE("fx: oversample must be 1/2/4, got %d -> 1", L); L = 1; }
    fx->L = L;
    if (L > 1) {
        int taps = 16 * L + 1;
        if (fir_init(&fx->up, taps, 0.5 / L, (double)L) != 0 ||
            fir_init(&fx->down, taps, 0.5 / L, 1.0) != 0) {
            bnn_fx_destroy(fx); return NULL;
        }
    }
    fx->phase = 0;
    return fx;
}

void bnn_fx_destroy(bnn_fx_t *fx) {
    if (!fx) return;
    fir_free(&fx->up);
    fir_free(&fx->down);
    bnn_free(fx);
}

void bnn_fx_reset(bnn_fx_t *fx) {
    if (!fx) return;
    if (fx->L > 1) { fir_reset(&fx->up); fir_reset(&fx->down); }
    fx->phase = 0;
}

int bnn_fx_process(bnn_fx_t *fx, const float *in, int n, float *out) {
    if (!fx || !in || !out || n < 0) return -1;
    const float mix = fx->cfg.mix, lvl = fx->cfg.out_level;
    if (fx->L <= 1) {
        for (int i = 0; i < n; ++i) {
            float sh = shape(fx, in[i]);
            out[i] = ((1.0f - mix) * in[i] + mix * sh) * lvl;
        }
        return 0;
    }
    const int L = fx->L;
    for (int i = 0; i < n; ++i) {
        float y_os = 0.0f;
        for (int p = 0; p < L; ++p) {
            float zin = (p == 0) ? in[i] : 0.0f;     /* 零插值 */
            float up = fir_step(&fx->up, zin);       /* 插值低通(增益 L) */
            float sh = shape(fx, up);                /* 过采样率整形 */
            float dn = fir_step(&fx->down, sh);      /* 抗混叠低通 */
            if (p == 0) y_os = dn;                   /* 抽取(取相位0) */
        }
        out[i] = ((1.0f - mix) * in[i] + mix * y_os) * lvl;
    }
    return 0;
}
