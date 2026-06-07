#ifndef BNN_TENSOR_H
#define BNN_TENSOR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BNN_TENSOR_MAX_DIM 4

typedef enum {
    BNN_DTYPE_F32 = 0,
    BNN_DTYPE_I8  = 1,
} bnn_dtype_t;

/*
 * 轻量张量:
 * - 形状最多 4 维 (N, C, H, W)
 * - 默认连续布局 (NCHW)
 * - 支持外部缓冲区 (owns=0) 避免拷贝
 * - 引用计数, 释放时减一
 */
typedef struct bnn_tensor {
    bnn_dtype_t dtype;
    int     ndim;
    int     shape[BNN_TENSOR_MAX_DIM];
    int     stride[BNN_TENSOR_MAX_DIM];  /* 元素数, 非字节 */
    size_t  numel;
    void   *data;
    int     owns;
    int     refcount;
} bnn_tensor_t;

/* 创建并分配数据 */
bnn_tensor_t *bnn_tensor_create(bnn_dtype_t dt, int ndim, const int *shape);
/* 用外部缓冲创建 (不拥有) */
bnn_tensor_t *bnn_tensor_wrap(bnn_dtype_t dt, int ndim, const int *shape, void *data);

bnn_tensor_t *bnn_tensor_retain(bnn_tensor_t *t);
void          bnn_tensor_release(bnn_tensor_t *t);

size_t bnn_tensor_elem_size(bnn_dtype_t dt);
void   bnn_tensor_fill_f32(bnn_tensor_t *t, float v);
void   bnn_tensor_copy_from(bnn_tensor_t *t, const void *src, size_t bytes);
void   bnn_tensor_zero(bnn_tensor_t *t);

#ifdef __cplusplus
}
#endif
#endif
