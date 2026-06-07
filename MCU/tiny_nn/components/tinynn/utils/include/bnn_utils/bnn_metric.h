#ifndef BNN_METRIC_H
#define BNN_METRIC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 分类精度: target 为 one-hot 或概率 */
float bnn_metric_accuracy(const float *logits, const float *target, int batch, int num_class);

/* 回归 MAE */
float bnn_metric_mae(const float *pred, const float *target, size_t numel);

/* 回归 RMSE */
float bnn_metric_rmse(const float *pred, const float *target, size_t numel);

#ifdef __cplusplus
}
#endif
#endif
