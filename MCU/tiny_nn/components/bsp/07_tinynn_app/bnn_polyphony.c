#include "bnn_polyphony.h"
#include <math.h>

#define PY_EPS 1e-10

void bnn_polyphony_analyze(const float *mag, int n_bins,
                           const bnn_poly_cfg_t *cfg, bnn_poly_result_t *out) {
    if (!out) return;
    out->n_peaks = 0; out->flatness = 0.0f; out->is_poly = 0;
    if (!mag || n_bins <= 2) return;

    bnn_poly_cfg_t c;
    if (cfg) c = *cfg; else bnn_poly_cfg_default(&c);

    /* 最大谱值 */
    float mx = 0.0f;
    for (int k = 0; k < n_bins; ++k) if (mag[k] > mx) mx = mag[k];
    float thr = c.peak_rel_thresh * mx;

    /* 显著谱峰计数 (局部极大且 > 阈值) */
    int n_peaks = 0;
    for (int k = 1; k < n_bins - 1; ++k) {
        if (mag[k] > thr && mag[k] >= mag[k - 1] && mag[k] > mag[k + 1]) n_peaks++;
    }

    /* 谱平坦度 = exp(mean(log p)) / mean(p), p = mag^2 */
    double log_sum = 0.0, lin_sum = 0.0;
    for (int k = 0; k < n_bins; ++k) {
        double p = (double)mag[k] * mag[k] + PY_EPS;
        log_sum += log(p);
        lin_sum += p;
    }
    double gmean = exp(log_sum / (double)n_bins);
    double amean = lin_sum / (double)n_bins;
    float flatness = (amean > PY_EPS) ? (float)(gmean / amean) : 0.0f;

    out->n_peaks = n_peaks;
    out->flatness = flatness;
    out->is_poly = (n_peaks >= c.peak_count_thresh) ? 1 : 0;
}
