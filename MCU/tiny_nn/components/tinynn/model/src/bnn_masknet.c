#include "bnn_model/bnn_masknet.h"
#include "bnn_frontend/bnn_specfront.h"
#include "bnn_synth/bnn_specsynth.h"
#include "bnn_graph/bnn_graph.h"
#include "bnn_graph/bnn_graph_ir.h"
#include "bnn_layer/bnn_layer.h"
#include "bnn_layer/bnn_xform_layers.h"
#include "bnn_utils/bnn_tensor.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <math.h>
#include <string.h>

struct bnn_masknet {
    bnn_mask_cfg_t cfg;
    int block_frames;     /* T */
    int cond_dim;
    int out_node;         /* head_mag 激活节点 (set_output) */
    int node_mask, node_phase, node_noise;  /* 三头激活节点 id */

    bnn_specfront_t *fe;
    bnn_graph_t     *g;
    bnn_specsynth_t *syn;

    float *emb_table; int emb_n, emb_dim;
    float *emb_cur;       /* [cond_dim] */

    /* scratch */
    float *in0;           /* [n_mels * T] */
    float *in1;           /* [cond_dim] */
    float *logmel;        /* [n_mels] */
    float *mag_blk;       /* [T * n_bins] */
    float *phase_blk;     /* [T * n_bins] */
    float *frame_tmp;     /* [n_fft] */
    float *mask_t, *dphi_t, *noise_t;  /* [n_mels]/[P]/[B] */
    float *hopbuf;        /* [hop] */
    int add_noise;
    float noise_gate_db;  /* 低于此输入帧能量则静音 (§6.5) */
};

static int build_graph(bnn_masknet_t *m) {
    const bnn_mask_cfg_t *c = &m->cfg;
    const int T = m->block_frames, E = m->cond_dim, k = c->kernel;
    const int Mn = c->n_mels, H = c->hidden;

    m->g = bnn_graph_create();
    if (!m->g) return -1;

    int sh0[3] = { 1, Mn, T };
    int in0 = bnn_graph_add_input(m->g, 3, sh0);
    int sh1[2] = { 1, E };
    int in1 = bnn_graph_add_input(m->g, 2, sh1);
    if (in0 == BNN_GRAPH_BAD_ID || in1 == BNN_GRAPH_BAD_ID) return -1;

    int pad1 = (k - 1) / 2;
    int pad2 = c->dilation2 * (k - 1) / 2;

    bnn_layer_cfg_t cc = {0};
    cc.in_channels = Mn; cc.out_channels = H; cc.kernel = k; cc.stride = 1; cc.padding = pad1; cc.dilation = 1;
    int c1 = bnn_graph_add_layer(m->g, "conv1d", &cc, &in0, 1);
    bnn_layer_cfg_t relu = {0}; relu.activation = 1;
    int a1 = bnn_graph_add_layer(m->g, "activation", &relu, &c1, 1);
    bnn_film_cfg_t fc1; fc1.channels = H; fc1.embedding_dim = E; fc1.gamma_plus_one = 0;
    bnn_layer_cfg_t lf1 = {0}; lf1.extra = &fc1;
    int d1[2] = { a1, in1 };
    int f1 = bnn_graph_add_layer(m->g, "film", &lf1, d1, 2);

    bnn_layer_cfg_t cc2 = {0};
    cc2.in_channels = H; cc2.out_channels = H; cc2.kernel = k; cc2.stride = 1;
    cc2.padding = pad2; cc2.dilation = c->dilation2;
    int c2 = bnn_graph_add_layer(m->g, "conv1d", &cc2, &f1, 1);
    int a2 = bnn_graph_add_layer(m->g, "activation", &relu, &c2, 1);
    bnn_film_cfg_t fc2; fc2.channels = H; fc2.embedding_dim = E; fc2.gamma_plus_one = 0;
    bnn_layer_cfg_t lf2 = {0}; lf2.extra = &fc2;
    int d2[2] = { a2, in1 };
    int f2 = bnn_graph_add_layer(m->g, "film", &lf2, d2, 2);

    bnn_layer_cfg_t cc3 = {0};
    cc3.in_channels = H; cc3.out_channels = Mn; cc3.kernel = k; cc3.stride = 1; cc3.padding = pad1; cc3.dilation = 1;
    int c3 = bnn_graph_add_layer(m->g, "conv1d", &cc3, &f2, 1);
    int a3 = bnn_graph_add_layer(m->g, "activation", &relu, &c3, 1);

    /* 三头: conv k1 + 缩放激活 */
    bnn_layer_cfg_t hm = {0}; hm.in_channels = Mn; hm.out_channels = Mn; hm.kernel = 1; hm.stride = 1;
    int hm_c = bnn_graph_add_layer(m->g, "conv1d", &hm, &a3, 1);
    bnn_layer_cfg_t am = {0}; am.activation = 2; am.param = c->gmax;          /* sigmoid×Gmax */
    int am_a = bnn_graph_add_layer(m->g, "activation", &am, &hm_c, 1);

    bnn_layer_cfg_t hp = {0}; hp.in_channels = Mn; hp.out_channels = c->phase_bands; hp.kernel = 1; hp.stride = 1;
    int hp_c = bnn_graph_add_layer(m->g, "conv1d", &hp, &a3, 1);
    bnn_layer_cfg_t ap = {0}; ap.activation = 3; ap.param = c->dphi_max;      /* tanh×Δφmax */
    int ap_a = bnn_graph_add_layer(m->g, "activation", &ap, &hp_c, 1);

    bnn_layer_cfg_t hn = {0}; hn.in_channels = Mn; hn.out_channels = c->noise_bands; hn.kernel = 1; hn.stride = 1;
    int hn_c = bnn_graph_add_layer(m->g, "conv1d", &hn, &a3, 1);
    bnn_layer_cfg_t an = {0}; an.activation = 4; an.param = 1.0f;             /* softplus */
    int an_a = bnn_graph_add_layer(m->g, "activation", &an, &hn_c, 1);

    if (c1 == BNN_GRAPH_BAD_ID || f1 == BNN_GRAPH_BAD_ID || c2 == BNN_GRAPH_BAD_ID ||
        f2 == BNN_GRAPH_BAD_ID || c3 == BNN_GRAPH_BAD_ID || am_a == BNN_GRAPH_BAD_ID ||
        ap_a == BNN_GRAPH_BAD_ID || an_a == BNN_GRAPH_BAD_ID) return -1;

    bnn_graph_set_output(m->g, am_a);
    m->out_node = am_a;
    m->node_mask = am_a; m->node_phase = ap_a; m->node_noise = an_a;
    return 0;
}

/* 数据驱动建图: 用图 IR 构建 CNN 主干, 输出头约定 [0]=mask [1]=phase [2]=noise. */
static int build_graph_ir(bnn_masknet_t *m, const void *ir_buf, size_t ir_len) {
    bnn_ir_model_t mdl;
    if (bnn_graph_build_from_ir(ir_buf, ir_len, &mdl) != 0) return -1;
    if (mdl.n_out < 3) {
        BNN_LOGE("masknet ir: 需 3 个输出头(mask/phase/noise), 实际 %d", mdl.n_out);
        bnn_graph_destroy(mdl.graph);
        return -1;
    }
    m->g = mdl.graph;
    m->node_mask  = mdl.out_nodes[0];
    m->node_phase = mdl.out_nodes[1];
    m->node_noise = mdl.out_nodes[2];
    m->out_node   = m->node_mask;
    return 0;
}

bnn_masknet_t *bnn_masknet_create(const bnn_mask_cfg_t *cfg, int num_instruments, int block_frames) {
    return bnn_masknet_create_ir(cfg, num_instruments, block_frames, NULL, 0);
}

bnn_masknet_t *bnn_masknet_create_ir(const bnn_mask_cfg_t *cfg, int num_instruments,
                                     int block_frames, const void *ir_buf, size_t ir_len) {
    if (!cfg || block_frames <= 0) { BNN_LOGE("masknet: bad args"); return NULL; }
    (void)num_instruments;
    bnn_layers_init_all();

    bnn_masknet_t *m = (bnn_masknet_t *)bnn_calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->cfg = *cfg;
    m->cfg.n_bins = cfg->n_fft / 2 + 1;
    m->block_frames = block_frames;
    m->cond_dim = BNN_MASK_COND_DIM(cfg);
    m->emb_dim = cfg->emb_dim;
    m->add_noise = 1;
    m->noise_gate_db = -60.0f;

    m->fe = bnn_specfront_create(&m->cfg);
    m->syn = bnn_specsynth_create(&m->cfg);
    if (!m->fe || !m->syn) { BNN_LOGE("masknet: front/synth fail"); goto fail; }
    /* 优先用图 IR (数据驱动); 失败或无 IR 回退内置硬编码图 */
    int gr = -1;
    if (ir_buf && ir_len > 0) {
        gr = build_graph_ir(m, ir_buf, ir_len);
        if (gr == 0) BNN_LOGI("masknet: 用图 IR 建图");
        else BNN_LOGI("masknet: 图 IR 建图失败, 回退内置硬编码图");
    }
    if (gr != 0) gr = build_graph(m);
    if (gr != 0) { BNN_LOGE("masknet: build graph fail"); goto fail; }

    int T = block_frames, Mn = cfg->n_mels, nb = m->cfg.n_bins;
    m->in0    = (float *)bnn_calloc((size_t)Mn * T, sizeof(float));
    m->in1    = (float *)bnn_calloc((size_t)m->cond_dim, sizeof(float));
    m->logmel = (float *)bnn_calloc((size_t)Mn, sizeof(float));
    m->mag_blk   = (float *)bnn_calloc((size_t)T * nb, sizeof(float));
    m->phase_blk = (float *)bnn_calloc((size_t)T * nb, sizeof(float));
    m->frame_tmp = (float *)bnn_calloc((size_t)cfg->n_fft, sizeof(float));
    m->mask_t  = (float *)bnn_calloc((size_t)Mn, sizeof(float));
    m->dphi_t  = (float *)bnn_calloc((size_t)cfg->phase_bands, sizeof(float));
    m->noise_t = (float *)bnn_calloc((size_t)cfg->noise_bands, sizeof(float));
    m->hopbuf  = (float *)bnn_calloc((size_t)cfg->hop, sizeof(float));
    m->emb_cur = (float *)bnn_calloc((size_t)m->cond_dim, sizeof(float));
    if (!m->in0 || !m->in1 || !m->logmel || !m->mag_blk || !m->phase_blk || !m->frame_tmp ||
        !m->mask_t || !m->dphi_t || !m->noise_t || !m->hopbuf || !m->emb_cur) goto fail;

    BNN_LOGI("masknet: T=%d n_mels=%d cond=%d params=%zu",
             T, Mn, m->cond_dim, bnn_graph_total_params(m->g));
    return m;
fail:
    bnn_masknet_destroy(m);
    return NULL;
}

void bnn_masknet_destroy(bnn_masknet_t *m) {
    if (!m) return;
    if (m->fe) bnn_specfront_destroy(m->fe);
    if (m->syn) bnn_specsynth_destroy(m->syn);
    if (m->g) bnn_graph_destroy(m->g);
    if (m->emb_table) bnn_free(m->emb_table);
    if (m->emb_cur) bnn_free(m->emb_cur);
    if (m->in0) bnn_free(m->in0);
    if (m->in1) bnn_free(m->in1);
    if (m->logmel) bnn_free(m->logmel);
    if (m->mag_blk) bnn_free(m->mag_blk);
    if (m->phase_blk) bnn_free(m->phase_blk);
    if (m->frame_tmp) bnn_free(m->frame_tmp);
    if (m->mask_t) bnn_free(m->mask_t);
    if (m->dphi_t) bnn_free(m->dphi_t);
    if (m->noise_t) bnn_free(m->noise_t);
    if (m->hopbuf) bnn_free(m->hopbuf);
    bnn_free(m);
}

int bnn_masknet_load_weights_mem(bnn_masknet_t *m, const void *buf, size_t nbytes) {
    if (!m || !buf) return -1;
    return bnn_graph_load_weights_mem(m->g, buf, nbytes);
}

void bnn_masknet_set_mel_norm(bnn_masknet_t *m, const float *mean, const float *std) {
    if (m) bnn_specfront_set_norm(m->fe, mean, std);
}

int bnn_masknet_set_embedding_table(bnn_masknet_t *m, const float *table, int n, int dim) {
    if (!m || !table || n <= 0) return -1;
    if (dim != m->emb_dim) { BNN_LOGE("masknet: emb dim %d != %d", dim, m->emb_dim); return -1; }
    if (m->emb_table) bnn_free(m->emb_table);
    m->emb_table = (float *)bnn_calloc((size_t)n * dim, sizeof(float));
    if (!m->emb_table) return -1;
    memcpy(m->emb_table, table, sizeof(float) * (size_t)n * dim);
    m->emb_n = n;
    return 0;
}

int bnn_masknet_set_instrument(bnn_masknet_t *m, int instrument_id) {
    if (!m || !m->emb_table) { BNN_LOGE("masknet: no embedding table"); return -1; }
    if (instrument_id < 0 || instrument_id >= m->emb_n) { BNN_LOGE("masknet: inst oob"); return -1; }
    memcpy(m->emb_cur, m->emb_table + (size_t)instrument_id * m->emb_dim,
           sizeof(float) * (size_t)m->emb_dim);
    return 0;
}

int bnn_masknet_set_embedding(bnn_masknet_t *m, const float *emb, int dim) {
    if (!m || !emb || dim != m->emb_dim) return -1;
    memcpy(m->emb_cur, emb, sizeof(float) * (size_t)dim);
    return 0;
}

void bnn_masknet_set_add_noise(bnn_masknet_t *m, int on) { if (m) m->add_noise = on ? 1 : 0; }
void bnn_masknet_set_smooth(bnn_masknet_t *m, float a) { if (m) bnn_specsynth_set_smooth(m->syn, a); }
void bnn_masknet_set_noise_gate(bnn_masknet_t *m, float db) { if (m) m->noise_gate_db = db; }
void bnn_masknet_reset(bnn_masknet_t *m) { if (m) bnn_specsynth_reset(m->syn); }

size_t bnn_masknet_num_params(const bnn_masknet_t *m) { return m ? bnn_graph_total_params(m->g) : 0; }
int    bnn_masknet_block_frames(const bnn_masknet_t *m) { return m ? m->block_frames : 0; }

/* 运行一块(已抽帧的 in0/mag_blk/phase_blk), 合成 chunk*hop 样点写到 sink 回调式 */
static int run_block(bnn_masknet_t *m, int chunk, int *gpos, int drop, int n_audio, float *out_audio) {
    const bnn_mask_cfg_t *c = &m->cfg;
    int T = m->block_frames, nb = c->n_bins, hop = c->hop;

    memcpy(m->in1, m->emb_cur, sizeof(float) * (size_t)m->cond_dim);
    if (bnn_graph_feed_input(m->g, 0, m->in0, 1) != 0) return -1;
    if (bnn_graph_feed_input(m->g, 1, m->in1, 1) != 0) return -1;
    if (!bnn_graph_forward(m->g)) { BNN_LOGE("masknet: forward fail"); return -1; }

    bnn_tensor_t *tm = bnn_graph_node_output(m->g, m->node_mask);   /* (1,n_mels,T) */
    bnn_tensor_t *tp = bnn_graph_node_output(m->g, m->node_phase);  /* (1,P,T) */
    bnn_tensor_t *tn = bnn_graph_node_output(m->g, m->node_noise);  /* (1,B,T) */
    if (!tm || !tp || !tn) return -1;
    const float *md = (const float *)tm->data;
    const float *pd = (const float *)tp->data;
    const float *nd = (const float *)tn->data;

    for (int t = 0; t < chunk; ++t) {
        for (int i = 0; i < c->n_mels; ++i) m->mask_t[i] = md[(size_t)i * T + t];
        for (int i = 0; i < c->phase_bands; ++i) m->dphi_t[i] = pd[(size_t)i * T + t];
        for (int i = 0; i < c->noise_bands; ++i) m->noise_t[i] = nd[(size_t)i * T + t];
        const float *mag = m->mag_blk + (size_t)t * nb;
        const float *phase = m->phase_blk + (size_t)t * nb;
        if (bnn_specsynth_process(m->syn, mag, phase, m->mask_t, m->dphi_t, m->noise_t,
                                  m->add_noise, m->hopbuf) != 0) return -1;
        /* 噪声门: 输入帧能量过低则该 hop 静音 (§6.5) */
        double e = 0.0;
        for (int k = 0; k < nb; ++k) e += (double)mag[k] * mag[k];
        float frms = (float)sqrt(e / (double)nb);
        if (20.0f * log10f(frms + 1e-8f) < m->noise_gate_db)
            for (int j = 0; j < hop; ++j) m->hopbuf[j] = 0.0f;
        for (int j = 0; j < hop; ++j) {
            int oi = (*gpos) + j - drop;
            if (oi >= 0 && oi < n_audio) out_audio[oi] = m->hopbuf[j];
        }
        *gpos += hop;
    }
    return 0;
}

int bnn_masknet_process_audio(bnn_masknet_t *m, const float *audio, int n_audio,
                              float *out_audio, int *out_n) {
    if (!m || !audio || !out_audio) return -1;
    const bnn_mask_cfg_t *c = &m->cfg;
    const int T = m->block_frames, n_fft = c->n_fft, hop = c->hop, nb = c->n_bins, Mn = c->n_mels;
    const int drop = n_fft / 2;                 /* center 前置零填充 */
    if (out_n) *out_n = 0;
    if (n_audio <= 0) return 0;

    int n_frames = n_audio / hop + 1;
    int gpos = 0;
    int chunk = 0;
    memset(m->in0, 0, sizeof(float) * (size_t)Mn * T);

    for (int t = 0; t < n_frames; ++t) {
        /* 取该帧 n_fft 样点 (center: padded index = t*hop+i, 真实 = -drop 偏移) */
        for (int i = 0; i < n_fft; ++i) {
            int ai = t * hop + i - drop;
            m->frame_tmp[i] = (ai >= 0 && ai < n_audio) ? audio[ai] : 0.0f;
        }
        bnn_specfront_extract(m->fe, m->frame_tmp, m->logmel,
                              m->mag_blk + (size_t)chunk * nb,
                              m->phase_blk + (size_t)chunk * nb);
        for (int cm = 0; cm < Mn; ++cm) m->in0[(size_t)cm * T + chunk] = m->logmel[cm];
        chunk++;

        if (chunk == T) {
            if (run_block(m, chunk, &gpos, drop, n_audio, out_audio) != 0) return -1;
            chunk = 0;
            memset(m->in0, 0, sizeof(float) * (size_t)Mn * T);
        }
    }
    if (chunk > 0) {
        if (run_block(m, chunk, &gpos, drop, n_audio, out_audio) != 0) return -1;
    }
    if (out_n) {
        int produced = gpos - drop;
        if (produced < 0) produced = 0;
        if (produced > n_audio) produced = n_audio;
        *out_n = produced;
    }
    return 0;
}
