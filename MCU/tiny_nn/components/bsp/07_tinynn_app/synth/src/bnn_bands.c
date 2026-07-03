#include "bnn_synth/bnn_bands.h"
#include <math.h>

/* HTK 公式, 与 features/mfcc.py 的 hz_to_mel / mel_to_hz 一致 */
static double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

int bnn_bands_make(int sample_rate, int fft_size, int n_bands,
                   float fmin, float fmax, int *edges_out) {
    if (sample_rate <= 0 || fft_size <= 1 || n_bands <= 0 || !edges_out) return -1;
    double fmax_hz = (fmax > 0.0f) ? (double)fmax : (double)sample_rate / 2.0;
    double fmin_hz = (double)fmin;
    int n_bins = fft_size / 2 + 1;

    double mmin = hz_to_mel(fmin_hz);
    double mmax = hz_to_mel(fmax_hz);
    for (int i = 0; i <= n_bands; ++i) {
        double mel = mmin + (mmax - mmin) * (double)i / (double)n_bands;  /* linspace */
        double hz  = mel_to_hz(mel);
        /* 与 numpy 一致: round-half-to-even (nearbyint 默认舍入模式) */
        int b = (int)nearbyint(hz / ((double)sample_rate / 2.0) * (double)(n_bins - 1));
        if (b < 0) b = 0;
        if (b > n_bins - 1) b = n_bins - 1;
        edges_out[i] = b;
    }
    /* 保证单调且每带至少 1 个 bin */
    for (int i = 1; i <= n_bands; ++i) {
        if (edges_out[i] <= edges_out[i - 1]) {
            int v = edges_out[i - 1] + 1;
            edges_out[i] = (v < n_bins - 1) ? v : (n_bins - 1);
        }
    }
    return n_bands + 1;
}
