#include "bnn_specmat.h"
#include <math.h>
#include <string.h>

#define SM_EPS 1e-8

/* HTK 公式, 与 features/mfcc.py 一致 */
static double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

int bnn_specmat_mel_basis(int sample_rate, int n_fft, int n_mels,
                          float fmin, float fmax, float *out) {
    if (sample_rate <= 0 || n_fft <= 1 || n_mels <= 0 || !out) return -1;
    int n_bins = n_fft / 2 + 1;
    double fmax_hz = (fmax > 0.0f) ? (double)fmax : (double)sample_rate / 2.0;
    double mmin = hz_to_mel((double)fmin), mmax = hz_to_mel(fmax_hz);

    /* hz_points = mel_to_hz(linspace(mmin, mmax, n_mels+2)) */
    /* fft_freqs[k] = k * (sr/2) / (n_bins-1) */
    memset(out, 0, sizeof(float) * (size_t)n_mels * n_bins);
    for (int m = 1; m <= n_mels; ++m) {
        double fl = mel_to_hz(mmin + (mmax - mmin) * (double)(m - 1) / (double)(n_mels + 1));
        double fc = mel_to_hz(mmin + (mmax - mmin) * (double)(m)     / (double)(n_mels + 1));
        double fr = mel_to_hz(mmin + (mmax - mmin) * (double)(m + 1) / (double)(n_mels + 1));
        if (fr <= fl) continue;
        float *row = out + (size_t)(m - 1) * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            double f = (double)k * ((double)sample_rate / 2.0) / (double)(n_bins - 1);
            double left  = (f - fl) / (fc - fl > 1e-9 ? fc - fl : 1e-9);
            double right = (fr - f) / (fr - fc > 1e-9 ? fr - fc : 1e-9);
            double v = left < right ? left : right;
            row[k] = (float)(v > 0.0 ? v : 0.0);
        }
    }
    return 0;
}

int bnn_specmat_mel_inv(const float *mel_basis, int n_mels, int n_bins, float *out) {
    if (!mel_basis || n_mels <= 0 || n_bins <= 0 || !out) return -1;
    for (int r = 0; r < n_bins; ++r) {
        double binsum = 0.0;
        for (int m = 0; m < n_mels; ++m) binsum += (double)mel_basis[(size_t)m * n_bins + r];
        double inv = 1.0 / (binsum + SM_EPS);
        for (int m = 0; m < n_mels; ++m)
            out[(size_t)r * n_mels + m] = (float)((double)mel_basis[(size_t)m * n_bins + r] * inv);
    }
    return 0;
}

int bnn_specmat_phase_inv(int n_bins, int phase_bands, float *out) {
    if (n_bins <= 0 || phase_bands <= 1 || !out) return -1;
    memset(out, 0, sizeof(float) * (size_t)n_bins * phase_bands);
    /* centers = linspace(0, n_bins-1, phase_bands) */
    for (int r = 0; r < n_bins; ++r) {
        double pos = (double)r * (double)(phase_bands - 1) / (double)(n_bins - 1); /* r 映射到 band 坐标 */
        if (pos <= 0.0) { out[(size_t)r * phase_bands + 0] = 1.0f; continue; }
        if (pos >= (double)(phase_bands - 1)) { out[(size_t)r * phase_bands + (phase_bands - 1)] = 1.0f; continue; }
        int j = (int)pos;
        if (j > phase_bands - 2) j = phase_bands - 2;
        double frac = pos - (double)j;
        out[(size_t)r * phase_bands + j]     = (float)(1.0 - frac);
        out[(size_t)r * phase_bands + j + 1] = (float)frac;
    }
    return 0;
}
