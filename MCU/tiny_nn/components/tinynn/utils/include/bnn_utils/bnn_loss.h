#ifndef BNN_LOSS_H
#define BNN_LOSS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 所有 loss 均返回 scalar loss; grad 写入 dL/dpred (shape 同 pred), 可为 NULL */
float bnn_loss_mse  (const float *pred, const float *target, float *grad, size_t numel);
float bnn_loss_l1   (const float *pred, const float *target, float *grad, size_t numel);
float bnn_loss_huber(const float *pred, const float *target, float *grad, size_t numel, float delta);

/* 二元交叉熵 (输入需为 [0,1] 概率, 例如 sigmoid 后) */
float bnn_loss_bce  (const float *pred, const float *target, float *grad, size_t numel);

/* logits + 交叉熵 (内部 softmax). target: one-hot 或概率 [batch, num_class] */
float bnn_loss_softmax_ce(const float *logits, const float *target,
                          float *grad, int batch, int num_class);

#ifdef __cplusplus
}
#endif
#endif
