#include "bnn_graph/bnn_graph.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include "bnn_utils/bnn_config.h"
#include <string.h>

#ifndef BNN_GRAPH_MAX_NODES
#define BNN_GRAPH_MAX_NODES 64
#endif
#ifndef BNN_GRAPH_MAX_DEPS
#define BNN_GRAPH_MAX_DEPS 2
#endif
#ifndef BNN_GRAPH_MAX_INPUTS
#define BNN_GRAPH_MAX_INPUTS 4
#endif

typedef struct {
    int           is_input;
    bnn_layer_t  *layer;            /* NULL 表示 input 节点 */
    int           deps[BNN_GRAPH_MAX_DEPS];
    int           n_dep;
    bnn_tensor_t *out;              /* 节点输出张量 (input 节点也用此持有) */
    float        *grad;             /* dL/d(out), 训练时按需分配 */
    int           shape[BNN_TENSOR_MAX_DIM];
    int           ndim;
} node_t;

struct bnn_graph {
    node_t nodes[BNN_GRAPH_MAX_NODES];
    int    n_nodes;
    int    inputs[BNN_GRAPH_MAX_INPUTS];
    int    n_inputs;
    int    output;
};

bnn_graph_t *bnn_graph_create(void) {
    bnn_graph_t *g = (bnn_graph_t *)bnn_calloc(1, sizeof(*g));
    if (g) g->output = BNN_GRAPH_BAD_ID;
    return g;
}

void bnn_graph_destroy(bnn_graph_t *g) {
    if (!g) return;
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *n = &g->nodes[i];
        /* layer 的 cached_output 已由 layer destroy 释放; 这里只 release input 节点拥有的 tensor */
        if (n->is_input && n->out) bnn_tensor_release(n->out);
        if (n->layer) n->layer->vtbl->destroy(n->layer);
        if (n->grad)  bnn_free(n->grad);
    }
    bnn_free(g);
}

int bnn_graph_add_input(bnn_graph_t *g, int ndim, const int *shape) {
    if (!g || g->n_nodes >= BNN_GRAPH_MAX_NODES) return BNN_GRAPH_BAD_ID;
    if (g->n_inputs >= BNN_GRAPH_MAX_INPUTS) return BNN_GRAPH_BAD_ID;
    int id = g->n_nodes++;
    node_t *n = &g->nodes[id];
    n->is_input = 1;
    n->ndim = ndim;
    for (int i = 0; i < ndim; ++i) n->shape[i] = shape[i];
    int s[BNN_TENSOR_MAX_DIM];
    memcpy(s, shape, sizeof(int) * ndim);
    if (s[0] <= 0) s[0] = 1;
    n->out = bnn_tensor_create(BNN_DTYPE_F32, ndim, s);
    g->inputs[g->n_inputs++] = id;
    return id;
}

int bnn_graph_add_layer(bnn_graph_t *g, const char *layer_name,
                        const bnn_layer_cfg_t *cfg,
                        const int *dep_ids, int n_dep) {
    if (!g || g->n_nodes >= BNN_GRAPH_MAX_NODES) return BNN_GRAPH_BAD_ID;
    if (n_dep <= 0 || n_dep > BNN_GRAPH_MAX_DEPS) return BNN_GRAPH_BAD_ID;
    bnn_layer_t *l = bnn_layer_create(layer_name, cfg);
    if (!l) return BNN_GRAPH_BAD_ID;
    int id = g->n_nodes++;
    node_t *n = &g->nodes[id];
    n->is_input = 0;
    n->layer = l;
    n->n_dep = n_dep;
    for (int i = 0; i < n_dep; ++i) n->deps[i] = dep_ids[i];
    /* 推断形状 (基于第 1 个依赖) */
    node_t *dep0 = &g->nodes[dep_ids[0]];
    l->vtbl->infer_shape(l, dep0->shape, dep0->ndim, n->shape, &n->ndim);
    return id;
}

void bnn_graph_set_output(bnn_graph_t *g, int node_id) {
    if (g) g->output = node_id;
}

int bnn_graph_feed_input(bnn_graph_t *g, int input_index, const float *data, int batch) {
    if (!g || input_index < 0 || input_index >= g->n_inputs) return -1;
    node_t *n = &g->nodes[g->inputs[input_index]];
    if (batch > 0 && n->shape[0] != batch) {
        n->shape[0] = batch;
        if (n->out) bnn_tensor_release(n->out);
        n->out = bnn_tensor_create(BNN_DTYPE_F32, n->ndim, n->shape);
        if (!n->out) return -1;
    }
    if (data) memcpy(n->out->data, data, n->out->numel * sizeof(float));
    return 0;
}

bnn_tensor_t *bnn_graph_forward(bnn_graph_t *g) {
    if (!g || g->output == BNN_GRAPH_BAD_ID) return NULL;
    bnn_tensor_t *ins[BNN_GRAPH_MAX_DEPS];
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *n = &g->nodes[i];
        if (n->is_input) continue;
        for (int j = 0; j < n->n_dep; ++j) ins[j] = g->nodes[n->deps[j]].out;
        bnn_tensor_t *y = n->layer->vtbl->forward(n->layer, ins, n->n_dep);
        if (!y) { BNN_LOGE("forward fail at node %d", i); return NULL; }
        /* 节点输出指针随 layer 的 cached_output (graph 不再额外持有) */
        n->out = y;
        /* 更新形状跟踪 */
        n->ndim = y->ndim;
        for (int k = 0; k < y->ndim; ++k) n->shape[k] = y->shape[k];
    }
    return g->nodes[g->output].out;
}

static float *ensure_grad(node_t *n) {
    size_t need = n->out ? n->out->numel : 0;
    if (need == 0) return NULL;
    if (!n->grad) n->grad = (float *)bnn_calloc(need, sizeof(float));
    return n->grad;
}

void bnn_graph_backward(bnn_graph_t *g, const float *dL_dout) {
    if (!g || g->output == BNN_GRAPH_BAD_ID) return;
    /* 清零所有节点 grad */
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *n = &g->nodes[i];
        if (n->grad && n->out) memset(n->grad, 0, n->out->numel * sizeof(float));
    }
    node_t *outn = &g->nodes[g->output];
    float *go = ensure_grad(outn);
    if (!go) return;
    memcpy(go, dL_dout, outn->out->numel * sizeof(float));

    bnn_tensor_t *ins[BNN_GRAPH_MAX_DEPS];
    float        *dxs[BNN_GRAPH_MAX_DEPS];
    /* 逆序遍历 */
    for (int i = g->n_nodes - 1; i >= 0; --i) {
        node_t *n = &g->nodes[i];
        if (n->is_input || !n->grad) continue;
        if (!n->layer->vtbl->backward) continue;
        for (int j = 0; j < n->n_dep; ++j) {
            ins[j] = g->nodes[n->deps[j]].out;
            dxs[j] = ensure_grad(&g->nodes[n->deps[j]]);
        }
        /* 由于依赖节点可能有多个消费者, 需累加: 让 layer 写到临时, 再 axpy. */
        /* 简化: 先暂存原 grad, 由 layer 直接写入, 再加回 */
        float *backup[BNN_GRAPH_MAX_DEPS] = {0};
        size_t bnums[BNN_GRAPH_MAX_DEPS] = {0};
        for (int j = 0; j < n->n_dep; ++j) {
            if (!dxs[j]) continue;
            bnums[j] = g->nodes[n->deps[j]].out->numel;
            backup[j] = (float *)bnn_malloc(bnums[j] * sizeof(float));
            if (backup[j]) memcpy(backup[j], dxs[j], bnums[j] * sizeof(float));
        }
        n->layer->vtbl->backward(n->layer, ins, n->n_dep, n->grad, dxs);
        for (int j = 0; j < n->n_dep; ++j) {
            if (!backup[j]) continue;
            for (size_t k = 0; k < bnums[j]; ++k) dxs[j][k] += backup[j][k];
            bnn_free(backup[j]);
        }
    }
}

void bnn_graph_collect_params(bnn_graph_t *g, bnn_optimizer_t *opt) {
    if (!g || !opt) return;
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *n = &g->nodes[i];
        if (n->is_input || !n->layer->vtbl->params) continue;
        bnn_layer_param_ref_t *p = n->layer->vtbl->params(n->layer);
        while (p) {
            bnn_optimizer_add_param(opt, p->data, p->grad, p->numel);
            p = p->next;
        }
    }
}

int bnn_graph_node_count(const bnn_graph_t *g) { return g ? g->n_nodes : 0; }

bnn_tensor_t *bnn_graph_node_output(const bnn_graph_t *g, int node_id) {
    if (!g || node_id < 0 || node_id >= g->n_nodes) return NULL;
    return g->nodes[node_id].out;
}

size_t bnn_graph_total_params(const bnn_graph_t *g) {
    if (!g) return 0;
    size_t total = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        const node_t *n = &g->nodes[i];
        if (n->is_input || !n->layer->vtbl->params) continue;
        bnn_layer_param_ref_t *p = n->layer->vtbl->params(n->layer);
        while (p) { total += p->numel; p = p->next; }
    }
    return total;
}

int bnn_graph_num_layers(const bnn_graph_t *g) {
    if (!g) return 0;
    int c = 0;
    for (int i = 0; i < g->n_nodes; ++i) if (!g->nodes[i].is_input) c++;
    return c;
}

#define BNN_WMAGIC 0x57574E42u  /* 'BNNW' */

#if BNN_ENABLE_FILE_IO
#include <stdio.h>

int bnn_graph_save_weights(const bnn_graph_t *g, const char *path) {
    if (!g || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { BNN_LOGE("save: cannot open %s", path); return -1; }
    unsigned int magic = BNN_WMAGIC;
    unsigned int ver = 1;
    unsigned long long n = (unsigned long long)bnn_graph_total_params(g);
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&ver,   sizeof(ver),   1, f);
    fwrite(&n,     sizeof(n),     1, f);
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *nd = (node_t *)&g->nodes[i];
        if (nd->is_input || !nd->layer->vtbl->params) continue;
        bnn_layer_param_ref_t *p = nd->layer->vtbl->params(nd->layer);
        while (p) {
            fwrite(p->data, sizeof(float), p->numel, f);
            p = p->next;
        }
    }
    fclose(f);
    BNN_LOGI("saved weights to %s (%llu floats)", path, n);
    return 0;
}

int bnn_graph_load_weights(const bnn_graph_t *g, const char *path) {
    if (!g || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) { BNN_LOGE("load: cannot open %s", path); return -1; }
    unsigned int magic = 0, ver = 0;
    unsigned long long n = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != BNN_WMAGIC) {
        BNN_LOGE("load: bad magic"); fclose(f); return -1;
    }
    if (fread(&ver, sizeof(ver), 1, f) != 1 || ver != 1) {
        BNN_LOGE("load: bad version"); fclose(f); return -1;
    }
    if (fread(&n, sizeof(n), 1, f) != 1) { fclose(f); return -1; }
    size_t expect = bnn_graph_total_params(g);
    if ((size_t)n != expect) {
        BNN_LOGE("load: param count mismatch %llu vs %zu", n, expect);
        fclose(f); return -1;
    }
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *nd = &g->nodes[i];
        if (nd->is_input || !nd->layer->vtbl->params) continue;
        bnn_layer_param_ref_t *p = nd->layer->vtbl->params(nd->layer);
        while (p) {
            if (fread(p->data, sizeof(float), p->numel, f) != p->numel) {
                BNN_LOGE("load: read fail"); fclose(f); return -1;
            }
            p = p->next;
        }
    }
    fclose(f);
    BNN_LOGI("loaded weights from %s", path);
    return 0;
}

#else  /* BNN_ENABLE_FILE_IO == 0: 无文件系统平台, 提供保持 ABI 的 stub */

int bnn_graph_save_weights(const bnn_graph_t *g, const char *path) {
    (void)g; (void)path;
    BNN_LOGE("save_weights unavailable (BNN_ENABLE_FILE_IO=0)");
    return -1;
}

int bnn_graph_load_weights(const bnn_graph_t *g, const char *path) {
    (void)g; (void)path;
    BNN_LOGE("load_weights unavailable (BNN_ENABLE_FILE_IO=0); use load_weights_mem");
    return -1;
}

#endif /* BNN_ENABLE_FILE_IO */

int bnn_graph_load_weights_mem(const bnn_graph_t *g, const void *buf, size_t nbytes) {
    if (!g || !buf) return -1;
    const unsigned char *p = (const unsigned char *)buf;
    size_t off = 0;
    if (nbytes < 16) { BNN_LOGE("load_mem: buffer too small"); return -1; }

    unsigned int magic = 0, ver = 0;
    unsigned long long n = 0;
    memcpy(&magic, p + off, sizeof(magic)); off += sizeof(magic);
    if (magic != BNN_WMAGIC) { BNN_LOGE("load_mem: bad magic"); return -1; }
    memcpy(&ver, p + off, sizeof(ver)); off += sizeof(ver);
    if (ver != 1) { BNN_LOGE("load_mem: bad version"); return -1; }
    memcpy(&n, p + off, sizeof(n)); off += sizeof(n);

    size_t expect = bnn_graph_total_params(g);
    if ((size_t)n != expect) {
        BNN_LOGE("load_mem: param count mismatch %llu vs %zu", n, expect);
        return -1;
    }
    if (nbytes < off + expect * sizeof(float)) {
        BNN_LOGE("load_mem: truncated buffer"); return -1;
    }
    for (int i = 0; i < g->n_nodes; ++i) {
        node_t *nd = (node_t *)&g->nodes[i];
        if (nd->is_input || !nd->layer->vtbl->params) continue;
        bnn_layer_param_ref_t *pr = nd->layer->vtbl->params(nd->layer);
        while (pr) {
            memcpy(pr->data, p + off, pr->numel * sizeof(float));
            off += pr->numel * sizeof(float);
            pr = pr->next;
        }
    }
    BNN_LOGI("loaded weights from memory (%llu floats)", n);
    return 0;
}

bnn_layer_t *bnn_graph_get_node_layer(const bnn_graph_t *g, int node_id)
{
    if (!g || node_id < 0 || node_id >= g->n_nodes) return NULL;
    const node_t *nd = &g->nodes[node_id];
    if (nd->is_input) return NULL;
    return nd->layer;
}
