#include "bnn_frontend/bnn_specfront.h"
#include "bnn_op/bnn_dsp.h"
#include "bnn_op/bnn_specmat.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct bnn_specfront {
    bnn_mask_cfg_t cfg;
    int    n_bins;
    float *window;     /* [n_fft] Hann */
    float *mel_basis;  /* [n_mels * n_bins] */
    float *mel_mean;   /* [n_mels] */
    float *mel_std;    /* [n_mels] */
    float *in_buf;     /* [n_fft] 加窗 */
    float *cplx;       /* [2*n_bins] rfft 输出 */
    float *mag_tmp;    /* [n_bins] */
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
    return fe;
}

void bnn_specfront_destroy(bnn_specfront_t *fe) {
    if (!fe) return;
    if (fe->window) bnn_free(fe->window);
    if (fe->mel_basis) bnn_free(fe->mel_basis);
    if (fe->mel_mean) bnn_free(fe->mel_mean);
    if (fe->mel_std) bnn_free(fe->mel_std);
    if (fe->in_buf) bnn_free(fe->in_buf);
    if (fe->cplx) bnn_free(fe->cplx);
    if (fe->mag_tmp) bnn_free(fe->mag_tmp);
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
        mg[k] = sqrtf(re * re + im * im);
        if (phase) phase[k] = atan2f(im, re);
    }
    if (mag) memcpy(mag, mg, sizeof(float) * (size_t)n_bins);

    if (logmel) {
        for (int m = 0; m < n_mels; ++m) {
            const float *row = fe->mel_basis + (size_t)m * n_bins;
            double e = 0.0;
            for (int k = 0; k < n_bins; ++k) e += (double)row[k] * mg[k];
            float lm = (float)log((double)e + (double)c->log_eps);
            float sd = fe->mel_std[m] != 0.0f ? fe->mel_std[m] : 1.0f;
            logmel[m] = (lm - fe->mel_mean[m]) / sd;
        }
    }
}
