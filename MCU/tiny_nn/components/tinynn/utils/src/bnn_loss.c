#include "bnn_utils/bnn_loss.h"
#include <math.h>
#include <float.h>

float bnn_loss_mse(const float *pred, const float *target, float *grad, size_t numel) {
    if (!pred || !target || numel == 0) return 0.f;
    double sum = 0.0;
    float inv = 1.0f / (float)numel;
    for (size_t i = 0; i < numel; ++i) {
        float d = pred[i] - target[i];
        sum += (double)d * d;
        if (grad) grad[i] = 2.f * d * inv;
    }
    return (float)(sum * inv);
}

float bnn_loss_l1(const float *pred, const float *target, float *grad, size_t numel) {
    if (!pred || !target || numel == 0) return 0.f;
    double sum = 0.0;
    float inv = 1.0f / (float)numel;
    for (size_t i = 0; i < numel; ++i) {
        float d = pred[i] - target[i];
        sum += fabsf(d);
        if (grad) grad[i] = (d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f)) * inv;
    }
    return (float)(sum * inv);
}

float bnn_loss_huber(const float *pred, const float *target, float *grad, size_t numel, float delta) {
    if (!pred || !target || numel == 0) return 0.f;
    double sum = 0.0;
    float inv = 1.f / (float)numel;
    for (size_t i = 0; i < numel; ++i) {
        float d = pred[i] - target[i];
        float ad = fabsf(d);
        if (ad <= delta) {
            sum += 0.5f * d * d;
            if (grad) grad[i] = d * inv;
        } else {
            sum += delta * (ad - 0.5f * delta);
            if (grad) grad[i] = (d > 0.f ? delta : -delta) * inv;
        }
    }
    return (float)(sum * inv);
}

float bnn_loss_bce(const float *pred, const float *target, float *grad, size_t numel) {
    if (!pred || !target || numel == 0) return 0.f;
    double sum = 0.0;
    float inv = 1.f / (float)numel;
    for (size_t i = 0; i < numel; ++i) {
        float p = pred[i] < 1e-7f ? 1e-7f : (pred[i] > 1.f - 1e-7f ? 1.f - 1e-7f : pred[i]);
        float t = target[i];
        sum -= t * log(p) + (1.f - t) * log(1.f - p);
        if (grad) grad[i] = ((p - t) / (p * (1.f - p))) * inv;
    }
    return (float)(sum * inv);
}

float bnn_loss_softmax_ce(const float *logits, const float *target,
                          float *grad, int batch, int num_class) {
    if (!logits || !target || batch <= 0 || num_class <= 0) return 0.f;
    double total = 0.0;
    for (int b = 0; b < batch; ++b) {
        const float *lg = logits + b * num_class;
        const float *tg = target + b * num_class;
        float mx = -FLT_MAX;
        for (int i = 0; i < num_class; ++i) if (lg[i] > mx) mx = lg[i];
        float sum = 0.f;
        if (grad) {
            float *gd = grad + b * num_class;
            for (int i = 0; i < num_class; ++i) { gd[i] = expf(lg[i] - mx); sum += gd[i]; }
            float inv = 1.f / sum;
            float loss = 0.f;
            for (int i = 0; i < num_class; ++i) {
                gd[i] *= inv;
                if (tg[i] > 0.f) loss -= tg[i] * logf(gd[i] + 1e-12f);
                gd[i] = (gd[i] - tg[i]) / (float)batch;
            }
            total += loss;
        } else {
            for (int i = 0; i < num_class; ++i) sum += expf(lg[i] - mx);
            float lse = logf(sum) + mx;
            float loss = 0.f;
            for (int i = 0; i < num_class; ++i) if (tg[i] > 0.f) loss -= tg[i] * (lg[i] - lse);
            total += loss;
        }
    }
    return (float)(total / batch);
}
