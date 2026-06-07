#ifndef BNN_DATALOADER_H
#define BNN_DATALOADER_H

#include "bnn_utils/bnn_dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const bnn_dataset_t *ds;
    int   batch;
    int   shuffle;
    int  *index;          /* 大小 = num */
    int   cursor;
    unsigned int rng;
} bnn_dataloader_t;

void bnn_dataloader_init(bnn_dataloader_t *dl, const bnn_dataset_t *ds, int batch, int shuffle, unsigned int seed);
void bnn_dataloader_free(bnn_dataloader_t *dl);
void bnn_dataloader_reset(bnn_dataloader_t *dl);  /* epoch 开始时调用 */

/* 取下一个 batch, 写入 x_out/y_out (调用方负责分配 batch*dim 大小).
 * 返回实际样本数, 0 表示 epoch 结束. */
int  bnn_dataloader_next(bnn_dataloader_t *dl, float *x_out, float *y_out);

#ifdef __cplusplus
}
#endif
#endif
