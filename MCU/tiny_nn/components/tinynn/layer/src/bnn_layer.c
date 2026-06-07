#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

#ifndef BNN_LAYER_MAX
#define BNN_LAYER_MAX 32
#endif

static const bnn_layer_factory_t *g_facs[BNN_LAYER_MAX];
static int                        g_fac_cnt = 0;

void bnn_layer_register(const bnn_layer_factory_t *f) {
    if (!f) return;
    /* 去重 */
    for (int i = 0; i < g_fac_cnt; ++i) if (g_facs[i] == f) return;
    if (g_fac_cnt >= BNN_LAYER_MAX) { BNN_LOGE("layer registry full"); return; }
    g_facs[g_fac_cnt++] = f;
    BNN_LOGD("layer registered: %s", f->name);
}

const bnn_layer_factory_t *bnn_layer_find(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_fac_cnt; ++i) {
        if (strcmp(g_facs[i]->name, name) == 0) return g_facs[i];
    }
    return NULL;
}

bnn_layer_t *bnn_layer_create(const char *name, const bnn_layer_cfg_t *cfg) {
    const bnn_layer_factory_t *f = bnn_layer_find(name);
    if (!f) { BNN_LOGE("layer not found: %s", name ? name : "(null)"); return NULL; }
    bnn_layer_t *l = f->vtbl->create(cfg);
    if (l) {
        /* 工厂统一回填 vtbl 与类型名: 各 layer 的 create 无需再自行设置 vtbl,
         * 也消除了 "create 内取不到自身 vtbl 地址" 的绕行写法 (见旧 residual). */
        l->vtbl = f->vtbl;
        l->type_name = f->name;
    }
    return l;
}

/* 平台无 constructor 时的兜底:
 * 通过 extern 声明所有内置 factory, 显式调用 register.
 * 在 GCC 平台 constructor 已注册过, register 内部去重. */
extern const bnn_layer_factory_t _bnn_fac_dense;
extern const bnn_layer_factory_t _bnn_fac_conv2d;
extern const bnn_layer_factory_t _bnn_fac_residual;
extern const bnn_layer_factory_t _bnn_fac_activation;
extern const bnn_layer_factory_t _bnn_fac_batchnorm1d;
extern const bnn_layer_factory_t _bnn_fac_pool2d;
extern const bnn_layer_factory_t _bnn_fac_flatten;
extern const bnn_layer_factory_t _bnn_fac_dropout;
extern const bnn_layer_factory_t _bnn_fac_softmax;
/* ESP-GT-XFORM 推理新增层 */
extern const bnn_layer_factory_t _bnn_fac_conv1d;
extern const bnn_layer_factory_t _bnn_fac_film;
extern const bnn_layer_factory_t _bnn_fac_embedding;

void bnn_layers_init_all(void) {
#if !(defined(__GNUC__) || defined(__clang__))
    bnn_layer_register(&_bnn_fac_dense);
    bnn_layer_register(&_bnn_fac_conv2d);
    bnn_layer_register(&_bnn_fac_residual);
    bnn_layer_register(&_bnn_fac_activation);
    bnn_layer_register(&_bnn_fac_batchnorm1d);
    bnn_layer_register(&_bnn_fac_pool2d);
    bnn_layer_register(&_bnn_fac_flatten);
    bnn_layer_register(&_bnn_fac_dropout);
    bnn_layer_register(&_bnn_fac_softmax);
    bnn_layer_register(&_bnn_fac_conv1d);
    bnn_layer_register(&_bnn_fac_film);
    bnn_layer_register(&_bnn_fac_embedding);
#endif
}
