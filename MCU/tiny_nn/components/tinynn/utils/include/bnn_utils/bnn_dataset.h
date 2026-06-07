#ifndef BNN_DATASET_H
#define BNN_DATASET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 极简内存数据集 (适合嵌入式小样本训练 / 推理) */
typedef struct bnn_dataset {
    const float *x;    /* [num, feat_dim] */
    const float *y;    /* [num, label_dim] */
    int num;
    int feat_dim;
    int label_dim;
} bnn_dataset_t;

/* 按 batch 取数据并拷贝到目标缓冲. 返回实际拷贝样本数. */
int bnn_dataset_get_batch(const bnn_dataset_t *ds, int start, int batch,
                          float *x_out, float *y_out);

/* 简单 min-max 归一化, in-place */
void bnn_preproc_minmax_f32(float *data, size_t numel, float min, float max);
/* 标准化: (x-mean)/std, in-place */
void bnn_preproc_standardize_f32(float *data, size_t numel, float mean, float std);

#ifdef __cplusplus
}
#endif
#endif
