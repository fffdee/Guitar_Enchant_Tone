#include "bnn_utils/bnn_tensor.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

size_t bnn_tensor_elem_size(bnn_dtype_t dt) {
    switch (dt) {
        case BNN_DTYPE_F32: return 4;
        case BNN_DTYPE_I8:  return 1;
    }
    return 0;
}

static void calc_strides(bnn_tensor_t *t) {
    int s = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        t->stride[i] = s;
        s *= t->shape[i];
    }
    t->numel = (size_t)s;
}

static bnn_tensor_t *alloc_header(bnn_dtype_t dt, int ndim, const int *shape) {
    if (ndim <= 0 || ndim > BNN_TENSOR_MAX_DIM) {
        BNN_LOGE("invalid ndim %d", ndim);
        return NULL;
    }
    bnn_tensor_t *t = (bnn_tensor_t *)bnn_calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->dtype = dt;
    t->ndim  = ndim;
    for (int i = 0; i < ndim; ++i) t->shape[i] = shape[i];
    calc_strides(t);
    t->refcount = 1;
    return t;
}

bnn_tensor_t *bnn_tensor_create(bnn_dtype_t dt, int ndim, const int *shape) {
    bnn_tensor_t *t = alloc_header(dt, ndim, shape);
    if (!t) return NULL;
    size_t bytes = t->numel * bnn_tensor_elem_size(dt);
    t->data = bnn_calloc(1, bytes);
    if (!t->data) { bnn_free(t); return NULL; }
    t->owns = 1;
    return t;
}

bnn_tensor_t *bnn_tensor_wrap(bnn_dtype_t dt, int ndim, const int *shape, void *data) {
    bnn_tensor_t *t = alloc_header(dt, ndim, shape);
    if (!t) return NULL;
    t->data = data;
    t->owns = 0;
    return t;
}

bnn_tensor_t *bnn_tensor_retain(bnn_tensor_t *t) {
    if (t) t->refcount++;
    return t;
}

void bnn_tensor_release(bnn_tensor_t *t) {
    if (!t) return;
    if (--t->refcount > 0) return;
    if (t->owns && t->data) bnn_free(t->data);
    bnn_free(t);
}

void bnn_tensor_fill_f32(bnn_tensor_t *t, float v) {
    if (!t || t->dtype != BNN_DTYPE_F32) return;
    float *p = (float *)t->data;
    for (size_t i = 0; i < t->numel; ++i) p[i] = v;
}

void bnn_tensor_copy_from(bnn_tensor_t *t, const void *src, size_t bytes) {
    if (!t || !src) return;
    size_t need = t->numel * bnn_tensor_elem_size(t->dtype);
    if (bytes > need) bytes = need;
    memcpy(t->data, src, bytes);
}

void bnn_tensor_zero(bnn_tensor_t *t) {
    if (!t) return;
    memset(t->data, 0, t->numel * bnn_tensor_elem_size(t->dtype));
}
