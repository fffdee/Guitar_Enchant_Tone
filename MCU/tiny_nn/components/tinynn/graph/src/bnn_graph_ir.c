#include "bnn_graph/bnn_graph_ir.h"
#include "bnn_layer/bnn_layer.h"
#include "bnn_layer/bnn_xform_layers.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

/* 小端读取 (不依赖结构体对齐, PC/MCU 一致) */
static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int32_t rd_i32(const uint8_t *p) { return (int32_t)rd_u32(p); }
static float   rd_f32(const uint8_t *p) { uint32_t u = rd_u32(p); float f; memcpy(&f, &u, 4); return f; }

/* Node 记录字段偏移 (见 bnn_graph_ir.h 布局) */
enum {
    O_TYPE = 0,           /* char[16] */
    O_KIND = 16,
    O_IN_FEATURES = 20, O_OUT_FEATURES = 24, O_IN_CH = 28, O_OUT_CH = 32,
    O_KERNEL = 36, O_STRIDE = 40, O_PAD = 44, O_DIL = 48, O_ACT = 52,
    O_PARAM = 56,         /* f32 */
    O_NDEP = 60,          /* input: ndim; layer: n_dep */
    O_A0 = 64, O_A1 = 68, O_A2 = 72, O_A3 = 76,
    O_EXTRA0 = 80,        /* film: gamma_plus_one */
};

int bnn_graph_build_from_ir(const void *buf, size_t nbytes, bnn_ir_model_t *out)
{
    if (!buf || !out || nbytes < 32) { BNN_LOGE("ir: bad args"); return -1; }
    memset(out, 0, sizeof(*out));
    const uint8_t *p = (const uint8_t *)buf;

    if (rd_u32(p) != BNN_IR_MAGIC) { BNN_LOGE("ir: bad magic"); return -1; }
    uint32_t ver  = rd_u32(p + 4);
    uint32_t nn   = rd_u32(p + 8);
    uint32_t nout = rd_u32(p + 12);
    uint32_t woff = rd_u32(p + 16);
    uint32_t wlen = rd_u32(p + 20);
    if (ver != BNN_IR_VERSION) { BNN_LOGE("ir: ver %u != %u", ver, BNN_IR_VERSION); return -1; }
    if (nout > BNN_IR_MAX_OUTPUTS) { BNN_LOGE("ir: too many outputs %u", nout); return -1; }

    size_t nodes_off = 32;
    size_t out_off   = nodes_off + (size_t)nn * BNN_IR_NODE_BYTES;
    if (out_off + (size_t)nout * 4 > nbytes) { BNN_LOGE("ir: truncated"); return -1; }

    bnn_layers_init_all();
    bnn_graph_t *g = bnn_graph_create();
    if (!g) return -1;

    /* IR 节点序号 -> 图节点 id 映射 */
    int *idmap = (int *)bnn_calloc((size_t)nn, sizeof(int));
    if (!idmap) { bnn_graph_destroy(g); return -1; }

    for (uint32_t i = 0; i < nn; ++i) {
        const uint8_t *e = p + nodes_off + (size_t)i * BNN_IR_NODE_BYTES;
        char type[17]; memcpy(type, e + O_TYPE, 16); type[16] = '\0';
        int32_t kind = rd_i32(e + O_KIND);
        int gid;

        if (kind == 0) {
            /* input: ndim + shape (a0..a3) */
            int ndim = rd_i32(e + O_NDEP);
            int shape[4] = { rd_i32(e + O_A0), rd_i32(e + O_A1), rd_i32(e + O_A2), rd_i32(e + O_A3) };
            if (ndim < 1 || ndim > 4) { BNN_LOGE("ir: bad ndim %d", ndim); goto fail; }
            gid = bnn_graph_add_input(g, ndim, shape);
        } else {
            bnn_layer_cfg_t cfg = {0};
            cfg.in_features  = rd_i32(e + O_IN_FEATURES);
            cfg.out_features = rd_i32(e + O_OUT_FEATURES);
            cfg.in_channels  = rd_i32(e + O_IN_CH);
            cfg.out_channels = rd_i32(e + O_OUT_CH);
            cfg.kernel       = rd_i32(e + O_KERNEL);
            cfg.stride       = rd_i32(e + O_STRIDE);
            cfg.padding      = rd_i32(e + O_PAD);
            cfg.dilation     = rd_i32(e + O_DIL);
            cfg.activation   = rd_i32(e + O_ACT);
            cfg.param        = rd_f32(e + O_PARAM);

            int ndep = rd_i32(e + O_NDEP);
            int draw[4] = { rd_i32(e + O_A0), rd_i32(e + O_A1), rd_i32(e + O_A2), rd_i32(e + O_A3) };
            if (ndep < 0 || ndep > 4) { BNN_LOGE("ir: bad ndep %d", ndep); goto fail; }
            int deps[4];
            for (int d = 0; d < ndep; ++d) {
                int ir_idx = draw[d];
                if (ir_idx < 0 || ir_idx >= (int)i) { BNN_LOGE("ir: dep %d 越界(节点%u)", ir_idx, i); goto fail; }
                deps[d] = idmap[ir_idx];   /* 拓扑序: 依赖必在前 */
            }

            /* film: 需要 gamma_plus_one, 通用 cfg 装不下, 走 extra 强类型 */
            bnn_film_cfg_t film;
            if (strcmp(type, "film") == 0) {
                film.channels       = cfg.out_channels;
                film.embedding_dim  = cfg.in_features;
                film.gamma_plus_one = rd_i32(e + O_EXTRA0);
                cfg.extra = &film;       /* 仅 add_layer 期间有效, create 内部会拷贝 */
            }
            gid = bnn_graph_add_layer(g, type, &cfg, deps, ndep);
        }

        if (gid == BNN_GRAPH_BAD_ID) { BNN_LOGE("ir: 节点%u '%s' 创建失败", i, type); goto fail; }
        idmap[i] = gid;
    }

    /* 输出节点 */
    out->n_out = (int)nout;
    for (uint32_t k = 0; k < nout; ++k) {
        int ir_idx = rd_i32(p + out_off + (size_t)k * 4);
        if (ir_idx < 0 || ir_idx >= (int)nn) { BNN_LOGE("ir: out idx %d 越界", ir_idx); goto fail; }
        out->out_nodes[k] = idmap[ir_idx];
    }
    if (nout > 0) bnn_graph_set_output(g, out->out_nodes[0]);

    /* 权重 (BNNW), 若 IR 内含 */
    if (wlen > 0) {
        if ((size_t)woff + wlen > nbytes) { BNN_LOGE("ir: weights 越界"); goto fail; }
        if (bnn_graph_load_weights_mem(g, p + woff, wlen) != 0) { BNN_LOGE("ir: 权重加载失败"); goto fail; }
    }

    bnn_free(idmap);
    out->graph = g;
    BNN_LOGI("ir: 建图成功 节点=%u 输出=%u params=%zu", nn, nout, bnn_graph_total_params(g));
    return 0;

fail:
    bnn_free(idmap);
    bnn_graph_destroy(g);
    out->graph = NULL;
    return -1;
}
