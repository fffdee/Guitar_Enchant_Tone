#ifndef BNN_LAYER_H
#define BNN_LAYER_H

#include "bnn_utils/bnn_tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bnn_layer;
typedef struct bnn_layer bnn_layer_t;

/* 通用层配置: 各 layer 自行解释. 嵌入式避免可变参数, 用统一 struct.
 *  - 通用字段覆盖绝大多数层;
 *  - dilation 为膨胀卷积常用项, 提升为通用字段;
 *  - extra 指向"层私有配置结构"(各 layer 自定义并强转), 提供无限扩展性而不污染通用结构.
 *    例: conv1d 的 bnn_conv1d_cfg_t、film 的 bnn_film_cfg_t、embedding 的 bnn_embedding_cfg_t. */
typedef struct bnn_layer_config {
    int   in_features;
    int   out_features;
    int   in_channels;
    int   out_channels;
    int   kernel;
    int   stride;
    int   padding;
    int   dilation;    /* 膨胀率, 默认 0/1 视为 1 */
    int   activation;  /* 0:none 1:relu 2:sigmoid 3:tanh */
    float param;       /* 通用浮点配置位 */
    const void *extra; /* 层私有配置指针 (可为 NULL); 由具体 layer 解释 */
} bnn_layer_cfg_t;

/* 层运行期参数引用 (供 optimizer 使用) */
typedef struct bnn_layer_param_ref {
    float *data;
    float *grad;
    size_t numel;
    struct bnn_layer_param_ref *next;
} bnn_layer_param_ref_t;

/* 虚函数表 (面向对象核心) */
typedef struct bnn_layer_vtbl {
    /* 构造: cfg 由用户提供, 返回 layer 实例 (含权重等) */
    bnn_layer_t *(*create)(const bnn_layer_cfg_t *cfg);
    /* 析构 */
    void  (*destroy)(bnn_layer_t *self);
    /* 推断输出形状, 由输入形状决定 */
    void  (*infer_shape)(bnn_layer_t *self, const int *in_shape, int in_ndim,
                         int *out_shape, int *out_ndim);
    /* 前向: 输入张量数组 -> 输出张量 (一般 1 输入 1 输出, residual 用 2 输入) */
    bnn_tensor_t *(*forward)(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in);
    /* 反向: dy = d L/d output; dx[i] 写入 d L/d inputs[i] (允许某项为 NULL 表示不需要) */
    void  (*backward)(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in,
                      const float *dy, float **dx);
    /* 注册可训练参数 */
    bnn_layer_param_ref_t *(*params)(bnn_layer_t *self);
} bnn_layer_vtbl_t;

/* 基类: 所有 layer 实例首字段必须是它 */
struct bnn_layer {
    const bnn_layer_vtbl_t *vtbl;
    const char  *type_name;
    /* 缓存: 输入与输出形状/张量 (供反向使用) */
    bnn_tensor_t *cached_input;
    bnn_tensor_t *cached_output;
};

/* ============ 静态注册 ============ */
typedef struct bnn_layer_factory {
    const char             *name;
    const bnn_layer_vtbl_t *vtbl;
} bnn_layer_factory_t;

void                       bnn_layer_register(const bnn_layer_factory_t *f);
const bnn_layer_factory_t *bnn_layer_find(const char *name);

/* 用户接口: 按名称创建一个层实例 */
bnn_layer_t               *bnn_layer_create(const char *name, const bnn_layer_cfg_t *cfg);

/*
 * 注册宏:
 *  - GCC/Clang 走 constructor 自动注册
 *  - 不支持 constructor 时, 用户需在 main 开头调用 bnn_layers_init_all()
 *    并在 bnn_layer_register_builtin.c 中列出所有内置 layer.
 */
#if defined(__GNUC__) || defined(__clang__)
#define BNN_REGISTER_LAYER(NAME, VTBL_PTR)                          \
    static const bnn_layer_factory_t _bnn_fac_##NAME = {            \
        .name = #NAME, .vtbl = (VTBL_PTR)                           \
    };                                                              \
    __attribute__((constructor)) static void _bnn_reg_##NAME(void) {\
        bnn_layer_register(&_bnn_fac_##NAME);                       \
    }
#else
/* 退路: 通过显式调用注册 (用户/平台层补充实现) */
#define BNN_REGISTER_LAYER(NAME, VTBL_PTR)                          \
    const bnn_layer_factory_t _bnn_fac_##NAME = {                   \
        .name = #NAME, .vtbl = (VTBL_PTR)                           \
    };
#endif

/* 对不支持 constructor 的平台, 用户在 main 起始调用此函数即可注册所有内置层 */
void bnn_layers_init_all(void);

#ifdef __cplusplus
}
#endif
#endif
