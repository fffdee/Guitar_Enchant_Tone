#include "bnn_utils/bnn_metric.h"
#include <math.h>
#include <float.h>

float bnn_metric_accuracy(const float *logits, const float *target, int batch, int num_class) {
    if (!logits || !target || batch <= 0 || num_class <= 0) return 0.f;
    int correct = 0;
    for (int b = 0; b < batch; ++b) {
        const float *p = logits + b * num_class;
        const float *t = target + b * num_class;
        int pi = 0, ti = 0;
        float pm = -FLT_MAX, tm = -FLT_MAX;
        for (int i = 0; i < num_class; ++i) {
            if (p[i] > pm) { pm = p[i]; pi = i; }
            if (t[i] > tm) { tm = t[i]; ti = i; }
        }
        if (pi == ti) correct++;
    }
    return (float)correct / (float)batch;
}

float bnn_metric_mae(const float *pred, const float *target, size_t numel) {
    if (!pred || !target || numel == 0) return 0.f;
    double s = 0;
    for (size_t i = 0; i < numel; ++i) s += fabsf(pred[i] - target[i]);
    return (float)(s / numel);
}
float bnn_metric_rmse(const float *pred, const float *target, size_t numel) {
    if (!pred || !target || numel == 0) return 0.f;
    double s = 0;
    for (size_t i = 0; i < numel; ++i) { float d = pred[i]-target[i]; s += d*d; }
    return (float)sqrt(s / numel);
}
