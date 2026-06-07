#include "bnn_utils/bnn_dataset.h"
#include <string.h>

int bnn_dataset_get_batch(const bnn_dataset_t *ds, int start, int batch,
                          float *x_out, float *y_out) {
    if (!ds) return 0;
    int n = batch;
    if (start + n > ds->num) n = ds->num - start;
    if (n <= 0) return 0;
    if (x_out) memcpy(x_out, ds->x + (size_t)start * ds->feat_dim,
                      (size_t)n * ds->feat_dim * sizeof(float));
    if (y_out) memcpy(y_out, ds->y + (size_t)start * ds->label_dim,
                      (size_t)n * ds->label_dim * sizeof(float));
    return n;
}

void bnn_preproc_minmax_f32(float *data, size_t numel, float min, float max) {
    if (!data || numel == 0) return;
    float range = max - min;
    if (range == 0.f) range = 1.f;
    for (size_t i = 0; i < numel; ++i) data[i] = (data[i] - min) / range;
}

void bnn_preproc_standardize_f32(float *data, size_t numel, float mean, float std) {
    if (!data || numel == 0) return;
    if (std == 0.f) std = 1.f;
    for (size_t i = 0; i < numel; ++i) data[i] = (data[i] - mean) / std;
}
