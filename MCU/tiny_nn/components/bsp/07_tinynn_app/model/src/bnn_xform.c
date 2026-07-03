#include "bnn_model/bnn_xform.h"
#include "bnn_frontend/bnn_frontend.h"
#include "bnn_synth/bnn_synth.h"
#include "bnn_graph/bnn_graph.h"
#include "bnn_layer/bnn_layer.h"
#include "bnn_layer/bnn_xform_layers.h"
#include "bnn_utils/bnn_tensor.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

struct bnn_xform {
    bnn_xform_cfg_t     cfg;
    bnn_xform_net_cfg_t net;
    int block_frames;   /* T */
    int in_ch;          /* feature_dim + embedding_dim */
    int out_dim;        /* n_harmonics + n_noise_bands */

    bnn_frontend_t *fe;
    bnn_graph_t    *g;
    bnn_synth_t    *synth;
    int             out_node;

    float *emb_table;   /* [emb_n * emb_dim], owned */
    int    emb_n, emb_dim;
    float *emb_cur;     /* [embedding_dim], owned */

    float *fmean, *fstd; /* [feature_dim], owned (NULL=不标准化) */

    /* scratch */
    float *in0;     /* [in_ch * T] 卷积输入 (特征 ⊕ 广播嵌入) */
    float *in1;     /* [embedding_dim] FiLM 条件向量 */
    float *f0;      /* [T] */
    float *feat;    /* [feature_dim] 逐帧 */
    float *params;  /* [T * out_dim] 逐帧, clamp 非负 */
};

void bnn_xform_net_cfg_default(bnn_xform_net_cfg_t *net) {
    if (!net) return;
    memset(net, 0, sizeof(*net));
    net->kernel = 3;
    net->n_conv = 4;
    net->channels[0] = 32; net->channels[1] = 64; net->channels[2] = 128; net->channels[3] = 64;
    net->dilations[0] = 1; net->dilations[1] = 2; net->dilations[2] = 4; net->dilations[3] = 1;
    net->film_mask = (1 << 1) | (1 << 2);  /* 第 2、3 个卷积后 FiLM */
    net->embedding_dim = 8;
    net->num_instruments = 4;
}

static int build_graph(bnn_xform_t *m) {
    const int T = m->block_frames;
    const int E = m->net.embedding_dim;
    const int k = m->net.kernel;

    m->g = bnn_graph_create();
    if (!m->g) return -1;

    int sh0[3] = { 1, m->in_ch, T };
    int in0 = bnn_graph_add_input(m->g, 3, sh0);
    int sh1[2] = { 1, E };
    int in1 = bnn_graph_add_input(m->g, 2, sh1);
    if (in0 == BNN_GRAPH_BAD_ID || in1 == BNN_GRAPH_BAD_ID) return -1;

    int prev = in0, prev_ch = m->in_ch;
    for (int i = 0; i < m->net.n_conv; ++i) {
        int dil = m->net.dilations[i] > 0 ? m->net.dilations[i] : 1;
        int pad = dil * (k - 1) / 2;     /* same padding (奇数核) */

        bnn_layer_cfg_t cc = {0};
        cc.in_channels = prev_ch; cc.out_channels = m->net.channels[i];
        cc.kernel = k; cc.stride = 1; cc.padding = pad; cc.dilation = dil;
        int cnode = bnn_graph_add_layer(m->g, "conv1d", &cc, &prev, 1);
        if (cnode == BNN_GRAPH_BAD_ID) return -1;

        int after = cnode;
        if (m->net.film_mask & (1 << i)) {
            bnn_film_cfg_t fc; fc.channels = m->net.channels[i]; fc.embedding_dim = E;
            fc.gamma_plus_one = 1;   /* DDSP: (1+gamma)*x+beta */
            bnn_layer_cfg_t lc = {0}; lc.extra = &fc;
            int deps[2] = { cnode, in1 };
            int fnode = bnn_graph_add_layer(m->g, "film", &lc, deps, 2);
            if (fnode == BNN_GRAPH_BAD_ID) return -1;
            after = fnode;
        }
        bnn_layer_cfg_t ac = {0}; ac.activation = 1;  /* relu */
        int anode = bnn_graph_add_layer(m->g, "activation", &ac, &after, 1);
        if (anode == BNN_GRAPH_BAD_ID) return -1;

        prev = anode; prev_ch = m->net.channels[i];
    }

    /* head: 1x1 conv 线性输出 out_dim */
    bnn_layer_cfg_t hc = {0};
    hc.in_channels = prev_ch; hc.out_channels = m->out_dim;
    hc.kernel = 1; hc.stride = 1; hc.padding = 0; hc.dilation = 1;
    int head = bnn_graph_add_layer(m->g, "conv1d", &hc, &prev, 1);
    if (head == BNN_GRAPH_BAD_ID) return -1;
    bnn_graph_set_output(m->g, head);
    m->out_node = head;
    return 0;
}

bnn_xform_t *bnn_xform_create(const bnn_xform_cfg_t *cfg,
                              const bnn_xform_net_cfg_t *net,
                              int block_frames) {
    if (!cfg || block_frames <= 0) { BNN_LOGE("xform: bad args"); return NULL; }

    /* 保证内置层已注册 (GCC constructor 已做; 其他工具链此调用补齐) */
    bnn_layers_init_all();

    bnn_xform_t *m = (bnn_xform_t *)bnn_calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->cfg = *cfg;
    if (net) m->net = *net; else bnn_xform_net_cfg_default(&m->net);
    m->block_frames = block_frames;
    m->in_ch  = cfg->feature_dim + m->net.embedding_dim;
    m->out_dim = cfg->n_harmonics + cfg->n_noise_bands;
    m->emb_dim = m->net.embedding_dim;

    m->fe = bnn_frontend_create(cfg);
    m->synth = bnn_synth_create(cfg);
    if (!m->fe || !m->synth) { BNN_LOGE("xform: frontend/synth create fail"); goto fail; }
    if (build_graph(m) != 0) { BNN_LOGE("xform: build graph fail"); goto fail; }

    const int T = block_frames, E = m->net.embedding_dim, F = cfg->feature_dim;
    m->in0    = (float *)bnn_calloc((size_t)m->in_ch * T, sizeof(float));
    m->in1    = (float *)bnn_calloc((size_t)E, sizeof(float));
    m->f0     = (float *)bnn_calloc((size_t)T, sizeof(float));
    m->feat   = (float *)bnn_calloc((size_t)F, sizeof(float));
    m->params = (float *)bnn_calloc((size_t)T * m->out_dim, sizeof(float));
    m->emb_cur = (float *)bnn_calloc((size_t)E, sizeof(float));
    if (!m->in0 || !m->in1 || !m->f0 || !m->feat || !m->params || !m->emb_cur) goto fail;

    BNN_LOGI("xform: T=%d in_ch=%d out_dim=%d params=%zu",
             T, m->in_ch, m->out_dim, bnn_graph_total_params(m->g));
    return m;

fail:
    bnn_xform_destroy(m);
    return NULL;
}

void bnn_xform_destroy(bnn_xform_t *m) {
    if (!m) return;
    if (m->fe) bnn_frontend_destroy(m->fe);
    if (m->synth) bnn_synth_destroy(m->synth);
    if (m->g) bnn_graph_destroy(m->g);
    if (m->emb_table) bnn_free(m->emb_table);
    if (m->emb_cur) bnn_free(m->emb_cur);
    if (m->fmean) bnn_free(m->fmean);
    if (m->fstd) bnn_free(m->fstd);
    if (m->in0) bnn_free(m->in0);
    if (m->in1) bnn_free(m->in1);
    if (m->f0) bnn_free(m->f0);
    if (m->feat) bnn_free(m->feat);
    if (m->params) bnn_free(m->params);
    bnn_free(m);
}

int bnn_xform_load_weights_mem(bnn_xform_t *m, const void *buf, size_t nbytes) {
    if (!m || !buf) return -1;
    return bnn_graph_load_weights_mem(m->g, buf, nbytes);
}

void bnn_xform_set_feature_norm(bnn_xform_t *m, const float *mean, const float *std) {
    if (!m) return;
    int F = m->cfg.feature_dim;
    if (!mean || !std) {
        if (m->fmean) { bnn_free(m->fmean); m->fmean = NULL; }
        if (m->fstd)  { bnn_free(m->fstd);  m->fstd  = NULL; }
        return;
    }
    if (!m->fmean) m->fmean = (float *)bnn_calloc((size_t)F, sizeof(float));
    if (!m->fstd)  m->fstd  = (float *)bnn_calloc((size_t)F, sizeof(float));
    if (m->fmean) memcpy(m->fmean, mean, sizeof(float) * (size_t)F);
    if (m->fstd)  memcpy(m->fstd,  std,  sizeof(float) * (size_t)F);
}

int bnn_xform_set_embedding_table(bnn_xform_t *m, const float *table, int n, int dim) {
    if (!m || !table || n <= 0) return -1;
    if (dim != m->emb_dim) { BNN_LOGE("xform: emb dim %d != %d", dim, m->emb_dim); return -1; }
    if (m->emb_table) bnn_free(m->emb_table);
    m->emb_table = (float *)bnn_calloc((size_t)n * dim, sizeof(float));
    if (!m->emb_table) return -1;
    memcpy(m->emb_table, table, sizeof(float) * (size_t)n * dim);
    m->emb_n = n;
    return 0;
}

int bnn_xform_set_instrument(bnn_xform_t *m, int instrument_id) {
    if (!m || !m->emb_table) { BNN_LOGE("xform: no embedding table"); return -1; }
    if (instrument_id < 0 || instrument_id >= m->emb_n) {
        BNN_LOGE("xform: instrument %d out of range [0,%d)", instrument_id, m->emb_n);
        return -1;
    }
    memcpy(m->emb_cur, m->emb_table + (size_t)instrument_id * m->emb_dim,
           sizeof(float) * (size_t)m->emb_dim);
    return 0;
}

int bnn_xform_set_embedding(bnn_xform_t *m, const float *emb, int dim) {
    if (!m || !emb) return -1;
    if (dim != m->emb_dim) { BNN_LOGE("xform: emb dim %d != %d", dim, m->emb_dim); return -1; }
    memcpy(m->emb_cur, emb, sizeof(float) * (size_t)dim);
    return 0;
}

void bnn_xform_reset(bnn_xform_t *m) { if (m && m->fe) bnn_frontend_reset(m->fe); }

int bnn_xform_block_frames(const bnn_xform_t *m) { return m ? m->block_frames : 0; }
size_t bnn_xform_num_params(const bnn_xform_t *m) { return m ? bnn_graph_total_params(m->g) : 0; }

int bnn_xform_process_frames(bnn_xform_t *m, const float *frames, int n_frames,
                             float *out_audio) {
    if (!m || !frames || !out_audio) return -1;
    const int T = m->block_frames;
    const int F = m->cfg.feature_dim;
    const int E = m->net.embedding_dim;
    const int hop = m->cfg.hop_size;
    const int frame_size = m->cfg.frame_size;
    if (n_frames <= 0 || n_frames > T) { BNN_LOGE("xform: n_frames %d not in [1,%d]", n_frames, T); return -1; }

    /* 清零输入块 (补零未用帧) */
    memset(m->in0, 0, sizeof(float) * (size_t)m->in_ch * T);
    memset(m->f0,  0, sizeof(float) * (size_t)T);

    /* 1) 逐帧前端 -> 特征(标准化) + f0; 写入卷积输入的特征通道 */
    for (int t = 0; t < n_frames; ++t) {
        bnn_frontend_extract(m->fe, frames + (size_t)t * frame_size, m->feat, &m->f0[t], NULL);
        if (m->fmean && m->fstd) bnn_frontend_standardize(&m->cfg, m->fmean, m->fstd, m->feat);
        for (int c = 0; c < F; ++c) m->in0[(size_t)c * T + t] = m->feat[c];
    }

    /* 2) 嵌入广播进卷积输入的嵌入通道 (channels F..F+E-1) */
    for (int c = 0; c < E; ++c) {
        float ev = m->emb_cur[c];
        float *ch = m->in0 + (size_t)(F + c) * T;
        for (int t = 0; t < T; ++t) ch[t] = ev;
    }

    /* 3) FiLM 条件向量 */
    memcpy(m->in1, m->emb_cur, sizeof(float) * (size_t)E);

    /* 4) 前向 */
    if (bnn_graph_feed_input(m->g, 0, m->in0, 1) != 0) return -1;
    if (bnn_graph_feed_input(m->g, 1, m->in1, 1) != 0) return -1;
    bnn_tensor_t *y = bnn_graph_forward(m->g);
    if (!y) { BNN_LOGE("xform: forward fail"); return -1; }

    /* 5) 输出 (1,out_dim,T) channel-major -> 逐帧 [n_frames][out_dim], clamp 非负 */
    const float *yd = (const float *)y->data;
    for (int t = 0; t < n_frames; ++t) {
        for (int p = 0; p < m->out_dim; ++p) {
            float v = yd[(size_t)p * T + t];
            m->params[(size_t)t * m->out_dim + p] = v > 0.0f ? v : 0.0f;
        }
    }

    /* 6) 合成 -> 音频 (只渲染有效帧对应的样点) */
    return bnn_synth_render(m->synth, m->f0, m->params, n_frames, out_audio, n_frames * hop);
}

int bnn_xform_process_audio(bnn_xform_t *m, const float *audio, int n_audio,
                            float *out_audio, int *out_n) {
    if (!m || !audio || !out_audio) return -1;
    const int T = m->block_frames;
    const int hop = m->cfg.hop_size;
    const int frame_size = m->cfg.frame_size;
    if (out_n) *out_n = 0;
    if (n_audio < frame_size) return 0;  /* 不足一帧 */

    int n_frames_total = (n_audio - frame_size) / hop + 1;

    /* 临时帧缓冲 (堆): 每块至多 T 帧, 各 frame_size 样点 */
    float *fbuf = (float *)bnn_malloc(sizeof(float) * (size_t)T * frame_size);
    if (!fbuf) { BNN_LOGE("xform: frame buf OOM"); return -1; }

    int produced = 0;
    for (int start = 0; start < n_frames_total; start += T) {
        int chunk = n_frames_total - start;
        if (chunk > T) chunk = T;
        for (int t = 0; t < chunk; ++t) {
            const float *src = audio + (size_t)(start + t) * hop;
            memcpy(fbuf + (size_t)t * frame_size, src, sizeof(float) * (size_t)frame_size);
        }
        int rc = bnn_xform_process_frames(m, fbuf, chunk, out_audio + (size_t)produced);
        if (rc != 0) { bnn_free(fbuf); return rc; }
        produced += chunk * hop;
    }
    bnn_free(fbuf);
    if (out_n) *out_n = produced;
    return 0;
}
