#include "bnn_layer/bnn_layer.h"
#include "bnn_layer/bnn_xform_layers.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

/*
 * Embedding 查表 (推理前向):
 *   inputs[0] = ids : [B, 1] (或 [B]); 元素为乐器/类别 id (以 float 存放, 取整)
 *   table     : [N, dim]
 *   输出 y    : [B, dim],  y[b] = table[id_b]
 *
 * 实时切换乐器时也可绕过本层, 直接把选定的 dim 维向量作为图输入喂给 film 的第二输入.
 * backward = NULL (训练在 PyTorch).
 */
typedef struct {
    bnn_layer_t base;
    int N, dim;
    float *table;   /* [N, dim] */
    bnn_layer_param_ref_t pref;
} embed_t;

static bnn_layer_t *embed_create(const bnn_layer_cfg_t *cfg) {
    if (!cfg || !cfg->extra) { BNN_LOGE("embedding: need extra(bnn_embedding_cfg_t)"); return NULL; }
    const bnn_embedding_cfg_t *ec = (const bnn_embedding_cfg_t *)cfg->extra;
    if (ec->num_embeddings <= 0 || ec->dim <= 0) { BNN_LOGE("embedding: invalid N/dim"); return NULL; }

    embed_t *e = (embed_t *)bnn_calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->N = ec->num_embeddings; e->dim = ec->dim;
    int n = e->N * e->dim;
    e->table = (float *)bnn_calloc(n, sizeof(float));
    if (!e->table) { bnn_free(e); return NULL; }
    e->pref.data = e->table; e->pref.grad = NULL; e->pref.numel = (size_t)n; e->pref.next = NULL;
    return &e->base;
}

static void embed_destroy(bnn_layer_t *self) {
    if (!self) return;
    embed_t *e = (embed_t *)self;
    if (e->table) bnn_free(e->table);
    if (self->cached_output) bnn_tensor_release(self->cached_output);
    bnn_free(e);
}

static void embed_infer(bnn_layer_t *self, const int *is, int in, int *os, int *on) {
    embed_t *e = (embed_t *)self; (void)in;
    os[0] = is[0]; os[1] = e->dim; *on = 2;
}

static bnn_tensor_t *embed_forward(bnn_layer_t *self, bnn_tensor_t **inputs, int n_in) {
    (void)n_in;
    embed_t *e = (embed_t *)self;
    bnn_tensor_t *ids = inputs[0];
    int B = ids->shape[0];
    int os[2] = { B, e->dim };

    if (self->cached_output && self->cached_output->shape[0] != B) {
        bnn_tensor_release(self->cached_output); self->cached_output = NULL;
    }
    if (!self->cached_output) {
        self->cached_output = bnn_tensor_create(BNN_DTYPE_F32, 2, os);
        if (!self->cached_output) return NULL;
    }
    self->cached_input = ids;

    const float *idp = (const float *)ids->data;
    int per = (int)(ids->numel / (size_t)(B > 0 ? B : 1));   /* 每行元素数(一般为1) */
    if (per < 1) per = 1;
    float *Y = (float *)self->cached_output->data;
    for (int bi = 0; bi < B; ++bi) {
        int id = (int)lroundf(idp[(size_t)bi * per]);
        if (id < 0) id = 0;
        if (id >= e->N) id = e->N - 1;
        memcpy(Y + (size_t)bi * e->dim, e->table + (size_t)id * e->dim, sizeof(float) * e->dim);
    }
    return self->cached_output;
}

static bnn_layer_param_ref_t *embed_params(bnn_layer_t *self) {
    return &((embed_t *)self)->pref;
}

static const bnn_layer_vtbl_t embed_vtbl = {
    .create = embed_create, .destroy = embed_destroy, .infer_shape = embed_infer,
    .forward = embed_forward, .backward = NULL, .params = embed_params,
};

BNN_REGISTER_LAYER(embedding, &embed_vtbl)
