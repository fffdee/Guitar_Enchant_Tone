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
#include "bnn_utils/bnn_perf.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

/* INT8 权重段格式常量 (与 PC 端 export.py _BNNW_I8_MAGIC 一致) */
#define BNNW_I8_MAGIC_VAL  0x3849574Eu
#define BNNW_I8_VERSION_VAL 1u

/* 非因果 CNN 左侧感受野半宽 (c1 pad1 + c2 dil2 pad2 + c3 pad1 = 4) */
#define BNN_MASKNET_CTX_FRAMES 4

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

    /* 诊断计时 (可选, 通过 bnn_masknet_set_tick_fn 注入平台计时函数) */
    int64_t (*tick_us)(void); /* 返回微秒的计时函数; NULL=不计时 */
    int64_t perf_specfront_us;
    int64_t perf_graph_us;
    int64_t perf_synth_us;
    int32_t perf_blocks;

    /* 推理进度 (供 status 查询) */
    int prog_frame;
    int prog_total;

    /* 跨块 logmel/谱上下文 (消除批次边界 CNN 零填充) */
    float *in0_ctx;       /* [n_mels * CTX] */
    float *mag_ctx;       /* [n_bins * CTX] */
    float *phase_ctx;     /* [2 * n_bins * CTX] */
    int    have_ctx;

    /* 调试: 首块首帧中间值打印 (xform debug) */
    int debug_dump;
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
    m->add_noise = 0;          /* 默认关闭噪声注入, 避免底噪; 可用 bnn_masknet_set_add_noise 开启 */
    m->noise_gate_db = -60.0f;

    /* 先建 specsynth (mel_inv 迁入 SRAM), 再建 specfront, 避免 SRAM 被占满 */
    m->syn = bnn_specsynth_create(&m->cfg);
    m->fe = bnn_specfront_create(&m->cfg);
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

    /* ── 性能优化: 标记 fuse_relu 和权重预取 ── */
    bnn_masknet_apply_perf_opts(m);

    int T = block_frames, Mn = cfg->n_mels, nb = m->cfg.n_bins;
    m->in0    = (float *)bnn_calloc((size_t)Mn * T, sizeof(float));
    m->in1    = (float *)bnn_calloc((size_t)m->cond_dim, sizeof(float));
    m->logmel = (float *)bnn_calloc((size_t)Mn, sizeof(float));
    m->mag_blk   = (float *)bnn_calloc((size_t)T * nb, sizeof(float));
    m->phase_blk = (float *)bnn_calloc((size_t)T * nb * 2, sizeof(float)); /* cos/sin 对 */
    m->frame_tmp = (float *)bnn_calloc((size_t)cfg->n_fft, sizeof(float));
    m->mask_t  = (float *)bnn_calloc((size_t)Mn, sizeof(float));
    m->dphi_t  = (float *)bnn_calloc((size_t)cfg->phase_bands, sizeof(float));
    m->noise_t = (float *)bnn_calloc((size_t)cfg->noise_bands, sizeof(float));
    m->hopbuf  = (float *)bnn_calloc((size_t)cfg->hop, sizeof(float));
    m->emb_cur = (float *)bnn_calloc((size_t)m->cond_dim, sizeof(float));
    if (!m->in0 || !m->in1 || !m->logmel || !m->mag_blk || !m->phase_blk || !m->frame_tmp ||
        !m->mask_t || !m->dphi_t || !m->noise_t || !m->hopbuf || !m->emb_cur) goto fail;

    m->in0_ctx = (float *)bnn_calloc((size_t)Mn * BNN_MASKNET_CTX_FRAMES, sizeof(float));
    m->mag_ctx = (float *)bnn_calloc((size_t)nb * BNN_MASKNET_CTX_FRAMES, sizeof(float));
    m->phase_ctx = (float *)bnn_calloc((size_t)nb * 2 * BNN_MASKNET_CTX_FRAMES, sizeof(float));
    if (!m->in0_ctx || !m->mag_ctx || !m->phase_ctx) goto fail;

    m->in0 = (float *)bnn_try_promote_internal(m->in0, sizeof(float) * (size_t)Mn * T);

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
    if (m->in0_ctx) bnn_free(m->in0_ctx);
    if (m->mag_ctx) bnn_free(m->mag_ctx);
    if (m->phase_ctx) bnn_free(m->phase_ctx);
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
void bnn_masknet_set_debug(bnn_masknet_t *m, int on) { if (m) m->debug_dump = on ? 1 : 0; }
void bnn_masknet_reset(bnn_masknet_t *m) {
    if (!m) return;
    bnn_specsynth_reset(m->syn);
    m->have_ctx = 0;
}

size_t bnn_masknet_num_params(const bnn_masknet_t *m) { return m ? bnn_graph_total_params(m->g) : 0; }
int    bnn_masknet_block_frames(const bnn_masknet_t *m) { return m ? m->block_frames : 0; }

/* center=True + reflect 边界 (与 PC np.pad(..., mode="reflect") 一致) */
static float reflect_sample(const float *audio, int n_audio, int ai)
{
    if (n_audio <= 0) return 0.0f;
    if (n_audio == 1) return audio[0];
    int idx = ai;
    while (idx < 0 || idx >= n_audio) {
        if (idx < 0) idx = -idx;
        if (idx >= n_audio) idx = 2 * (n_audio - 1) - idx;
    }
    return audio[idx];
}

static void save_block_ctx(bnn_masknet_t *m, int T)
{
    const int Mn = m->cfg.n_mels, nb = m->cfg.n_bins;
    const int C = BNN_MASKNET_CTX_FRAMES;
    const int base = T - C;
    for (int t = 0; t < C; ++t) {
        int col = base + t;
        for (int cm = 0; cm < Mn; ++cm)
            m->in0_ctx[(size_t)cm * C + t] = m->in0[(size_t)cm * T + col];
        memcpy(m->mag_ctx + (size_t)t * nb, m->mag_blk + (size_t)col * nb,
               sizeof(float) * (size_t)nb);
        memcpy(m->phase_ctx + (size_t)t * nb * 2, m->phase_blk + (size_t)col * nb * 2,
               sizeof(float) * (size_t)nb * 2);
    }
    m->have_ctx = 1;
}

static void prepend_block_ctx(bnn_masknet_t *m, int T)
{
    const int Mn = m->cfg.n_mels, nb = m->cfg.n_bins;
    const int C = BNN_MASKNET_CTX_FRAMES;
    for (int t = 0; t < C; ++t) {
        for (int cm = 0; cm < Mn; ++cm)
            m->in0[(size_t)cm * T + t] = m->in0_ctx[(size_t)cm * C + t];
        memcpy(m->mag_blk + (size_t)t * nb, m->mag_ctx + (size_t)t * nb,
               sizeof(float) * (size_t)nb);
        memcpy(m->phase_blk + (size_t)t * nb * 2, m->phase_ctx + (size_t)t * nb * 2,
               sizeof(float) * (size_t)nb * 2);
    }
}

static void debug_log_block0_frame0(bnn_masknet_t *m)
{
    const int Mn = m->cfg.n_mels, nb = m->cfg.n_bins;
    float gain[8];
    bnn_specsynth_peek_gain_lin(m->syn, m->mask_t, gain, 8);
    BNN_LOGI("debug block0 frame0 logmel[0:8]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             m->logmel[0], m->logmel[1], m->logmel[2], m->logmel[3],
             m->logmel[4], m->logmel[5], m->logmel[6], m->logmel[7]);
    BNN_LOGI("debug block0 frame0 mask[0:8]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             m->mask_t[0], m->mask_t[1], m->mask_t[2], m->mask_t[3],
             m->mask_t[4], m->mask_t[5], m->mask_t[6], m->mask_t[7]);
    BNN_LOGI("debug block0 frame0 gain_lin[0:8]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             gain[0], gain[1], gain[2], gain[3], gain[4], gain[5], gain[6], gain[7]);
    (void)Mn; (void)nb;
}

/* 运行一块(已抽帧的 in0/mag_blk/phase_blk), 合成 chunk*hop 样点写到 sink 回调式.
 * syn_start: 从第几帧开始 ISTFT 输出 (有跨块上下文时跳过前 C 帧, 避免重复合成). */
static int run_block(bnn_masknet_t *m, int chunk, int syn_start, int *gpos, int drop,
                     int n_audio, float *out_audio) {
    const bnn_mask_cfg_t *c = &m->cfg;
    int T = m->block_frames, nb = c->n_bins, hop = c->hop;

    /* 末块不足 T 帧: 尾部清零, 避免 CNN 读到上一块残留 */
    if (chunk < T) {
        const int Mn = c->n_mels;
        for (int cm = 0; cm < Mn; ++cm)
            memset(m->in0 + (size_t)cm * T + chunk, 0,
                   (size_t)(T - chunk) * sizeof(float));
    }

    memcpy(m->in1, m->emb_cur, sizeof(float) * (size_t)m->cond_dim);
    if (bnn_graph_feed_input(m->g, 0, m->in0, 1) != 0) return -1;
    if (bnn_graph_feed_input(m->g, 1, m->in1, 1) != 0) return -1;

    int64_t t_g0 = m->tick_us ? m->tick_us() : 0;
    if (!bnn_graph_forward(m->g)) { BNN_LOGE("masknet: forward fail"); return -1; }
    if (m->tick_us) m->perf_graph_us += m->tick_us() - t_g0;

    bnn_tensor_t *tm = bnn_graph_node_output(m->g, m->node_mask);   /* (1,n_mels,T) */
    bnn_tensor_t *tp = bnn_graph_node_output(m->g, m->node_phase);  /* (1,P,T) */
    bnn_tensor_t *tn = bnn_graph_node_output(m->g, m->node_noise);  /* (1,B,T) */
    if (!tm || !tp || !tn) return -1;
    const float *md = (const float *)tm->data;
    const float *pd = (const float *)tp->data;
    const float *nd = (const float *)tn->data;

    if (syn_start < 0) syn_start = 0;
    if (syn_start > chunk) syn_start = chunk;

    int64_t t_syn0 = m->tick_us ? m->tick_us() : 0;
    for (int t = syn_start; t < chunk; ++t) {
        for (int i = 0; i < c->n_mels; ++i) m->mask_t[i] = md[(size_t)i * T + t];
        for (int i = 0; i < c->phase_bands; ++i) m->dphi_t[i] = pd[(size_t)i * T + t];
        for (int i = 0; i < c->noise_bands; ++i) m->noise_t[i] = nd[(size_t)i * T + t];

        if (m->debug_dump && m->perf_blocks == 0 && t == 0) {
            for (int cm = 0; cm < c->n_mels; ++cm)
                m->logmel[cm] = m->in0[(size_t)cm * T + t];
            debug_log_block0_frame0(m);
            m->debug_dump = 0;
        }

        /* 首块第 0 帧: 打印 mask/noise 统计, 用于诊断 CNN 是否生效 */
        if (m->perf_blocks == 0 && t == 0) {
            float mmin = m->mask_t[0], mmax = m->mask_t[0], msum = 0.0f;
            for (int i = 0; i < c->n_mels; ++i) {
                float v = m->mask_t[i];
                if (v < mmin) mmin = v;
                if (v > mmax) mmax = v;
                msum += v;
            }
            float mmean = msum / (float)c->n_mels;
            float nmin = m->noise_t[0], nmax = m->noise_t[0], nsum = 0.0f;
            for (int i = 0; i < c->noise_bands; ++i) {
                float v = m->noise_t[i];
                if (v < nmin) nmin = v;
                if (v > nmax) nmax = v;
                nsum += v;
            }
            float nmean = nsum / (float)c->noise_bands;
            BNN_LOGI("mask[%d]: min=%.3f max=%.3f mean=%.3f  (正常 bass 约 3.0~4.0, 失效≈2.0)",
                     c->n_mels, mmin, mmax, mmean);
            BNN_LOGI("noise_t[%d]: min=%.3f max=%.3f mean=%.3f  add_noise=%d",
                     c->noise_bands, nmin, nmax, nmean, m->add_noise);
            if (mmax - mmin < 0.05f && mmean > 1.8f && mmean < 2.2f)
                BNN_LOGE("mask 似恒等 (~2.0): CNN 未生效, 检查 xform_model.bin 或禁用 INT8");
            if (nmax - nmin < 1e-4f && fabsf(nmean - 0.693147f) < 1e-3f)
                BNN_LOGE("noise_t 全 0.693: CNN backbone 输出≈0");
        }

        const float *mag = m->mag_blk + (size_t)t * nb;
        const float *phase = m->phase_blk + (size_t)t * nb * 2;
        if (bnn_specsynth_process(m->syn, mag, phase, m->mask_t, m->dphi_t, m->noise_t,
                                  m->add_noise, m->hopbuf) != 0) return -1;
        /* 噪声门: 输入帧能量过低则该 hop 静音 (§6.5) */
        float e = 0.0f;
        for (int k = 0; k < nb; ++k) e += mag[k] * mag[k];
        float frms = sqrtf(e / (float)nb);
        if (20.0f * log10f(frms + 1e-8f) < m->noise_gate_db)
            for (int j = 0; j < hop; ++j) m->hopbuf[j] = 0.0f;
        for (int j = 0; j < hop; ++j) {
            int oi = (*gpos) + j - drop;
            if (oi >= 0 && oi < n_audio) out_audio[oi] = m->hopbuf[j];
        }
        *gpos += hop;
    }
    if (m->tick_us) m->perf_synth_us += m->tick_us() - t_syn0;
    m->perf_blocks++;
    return 0;
}

int bnn_masknet_process_audio(bnn_masknet_t *m, const float *audio, int n_audio,
                              float *out_audio, int *out_n) {
    if (!m || !audio || !out_audio) return -1;
    const bnn_mask_cfg_t *c = &m->cfg;
    const int T = m->block_frames, n_fft = c->n_fft, hop = c->hop, nb = c->n_bins, Mn = c->n_mels;
    const int C = BNN_MASKNET_CTX_FRAMES;
    const int drop = n_fft / 2;                 /* center 前置零填充 */
    if (out_n) *out_n = 0;
    if (n_audio <= 0) return 0;

    int n_frames = n_audio / hop + 1;
    int gpos = 0;
    int chunk = 0;
    m->prog_frame = 0;
    m->prog_total = n_frames;
    m->have_ctx = 0;
    memset(m->in0, 0, sizeof(float) * (size_t)Mn * T);

    for (int t = 0; t < n_frames; ++t) {
        m->prog_frame = t + 1;
        /* 取该帧 n_fft 样点 (center + reflect, 与 PC stft 一致) */
        for (int i = 0; i < n_fft; ++i) {
            int ai = t * hop + i - drop;
            m->frame_tmp[i] = reflect_sample(audio, n_audio, ai);
        }
        int write_col = m->have_ctx ? (C + chunk) : chunk;
        int64_t t_fe0 = m->tick_us ? m->tick_us() : 0;
        bnn_specfront_extract(m->fe, m->frame_tmp, m->logmel,
                              m->mag_blk + (size_t)write_col * nb,
                              m->phase_blk + (size_t)write_col * nb * 2);
        if (m->tick_us) m->perf_specfront_us += m->tick_us() - t_fe0;
        for (int cm = 0; cm < Mn; ++cm)
            m->in0[(size_t)cm * T + write_col] = m->logmel[cm];
        chunk++;

        int target = m->have_ctx ? (T - C) : T;
        if (chunk == target) {
            if (m->have_ctx) prepend_block_ctx(m, T);
            int syn_start = m->have_ctx ? C : 0;
            if (run_block(m, T, syn_start, &gpos, drop, n_audio, out_audio) != 0) return -1;
            save_block_ctx(m, T);
            chunk = 0;
        }
    }
    if (chunk > 0) {
        int cols = m->have_ctx ? (C + chunk) : chunk;
        if (m->have_ctx) prepend_block_ctx(m, T);
        int syn_start = m->have_ctx ? C : 0;
        if (run_block(m, cols, syn_start, &gpos, drop, n_audio, out_audio) != 0) return -1;
    }
    if (out_n) {
        int produced = gpos - drop;
        if (produced < 0) produced = 0;
        if (produced > n_audio) produced = n_audio;
        *out_n = produced;
    }
    return 0;
}

/* ── INT8 权重加载 ────────────────────────────────────────────────────────── */

static inline uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

int bnn_masknet_load_weights_i8_mem(bnn_masknet_t *m, const void *buf, size_t nbytes)
{
    if (!m || !m->g || !buf || nbytes < 16) return 0;  /* 静默跳过 */

    const uint8_t *p   = (const uint8_t *)buf;
    const uint8_t *end = p + nbytes;

    /* 验证内部魔数 */
    if (rd32(p) != BNNW_I8_MAGIC_VAL) {
        BNN_LOGE("masknet i8: bad magic %08" PRIx32, rd32(p));
        return -1;
    }
    if (rd32(p + 4) != BNNW_I8_VERSION_VAL) {
        BNN_LOGE("masknet i8: unsupported version %" PRIu32, rd32(p + 4));
        return -1;
    }
    uint32_t num_conv = rd32(p + 8);
    p += 16;  /* 跳过 16B 头 */

    int injected = 0;
    int n_nodes  = bnn_graph_node_count(m->g);

    /* 遍历图节点, 按拓扑顺序为 conv1d 层注入 INT8 权重 */
    for (int ni = 0; ni < n_nodes && num_conv > 0; ++ni) {
        bnn_layer_t *layer = bnn_graph_get_node_layer(m->g, ni);
        if (!layer || !layer->type_name) continue;
        if (strcmp(layer->type_name, "conv1d") != 0) continue;

        /* 解析下一层块头: Cout(u32) Cin_K(u32) nbytes_w(u32) _rsv(u32) */
        if (p + 16 > end) { BNN_LOGE("masknet i8: truncated at layer %d", injected); return -1; }
        uint32_t Cout   = rd32(p);
        uint32_t Cin_K  = rd32(p + 4);
        uint32_t nbytes_w = rd32(p + 8);
        p += 16;

        /* INT8 权重 */
        if (p + nbytes_w > end) { BNN_LOGE("masknet i8: W_i8 OOB layer %d", injected); return -1; }
        const int8_t *W_i8 = (const int8_t *)p;
        p += nbytes_w;

        /* 4 字节对齐填充 */
        uint32_t pad = (4u - (nbytes_w % 4u)) % 4u;
        p += pad;

        /* float32 scale [Cout] */
        size_t scale_sz = sizeof(float) * Cout;
        if (p + scale_sz > end) { BNN_LOGE("masknet i8: scale OOB layer %d", injected); return -1; }
        const float *scale = (const float *)p;
        p += scale_sz;

        /* 注入到 conv1d 层 */
        if (conv1d_set_weights_i8(layer, W_i8, scale, (int)Cout, (int)Cin_K) == 0) {
            injected++;
        } else {
            BNN_LOGI("masknet i8: set_weights_i8 failed layer %d", injected);
        }
        num_conv--;
    }

    if (injected > 0) {
        BNN_LOGI("masknet i8: injected %d conv1d layers", injected);
    }
    return injected;
}

/* ── 诊断计时 API ─────────────────────────────────────────────────────────── */

void bnn_masknet_set_tick_fn(bnn_masknet_t *m, int64_t (*fn)(void))
{
    if (m) m->tick_us = fn;
}

void bnn_masknet_perf_reset(bnn_masknet_t *m)
{
    if (!m) return;
    m->perf_specfront_us = 0;
    m->perf_graph_us     = 0;
    m->perf_synth_us     = 0;
    m->perf_blocks       = 0;
}

void bnn_masknet_get_progress(const bnn_masknet_t *m, int *frame, int *total)
{
    if (!m) return;
    if (frame) *frame = m->prog_frame;
    if (total) *total = m->prog_total;
}

void bnn_masknet_perf_log(const bnn_masknet_t *m)
{
    if (!m || !m->tick_us) {
        BNN_LOGI("masknet perf: 未设置计时函数, 调用 bnn_masknet_set_tick_fn()");
        return;
    }
    int64_t total = m->perf_specfront_us + m->perf_graph_us + m->perf_synth_us;
    int32_t nb    = m->perf_blocks > 0 ? m->perf_blocks : 1;

    BNN_LOGI("masknet perf (%d blocks):", (int)nb);
    BNN_LOGI("  specfront : %lld ms  (%lld ms/block)",
             (long long)(m->perf_specfront_us / 1000),
             (long long)(m->perf_specfront_us / 1000 / nb));
    BNN_LOGI("  graph fwd : %lld ms  (%lld ms/block)",
             (long long)(m->perf_graph_us / 1000),
             (long long)(m->perf_graph_us / 1000 / nb));
    BNN_LOGI("  specsynth : %lld ms  (%lld ms/block)",
             (long long)(m->perf_synth_us / 1000),
             (long long)(m->perf_synth_us / 1000 / nb));
    BNN_LOGI("  total     : %lld ms", (long long)(total / 1000));
}

/* ============================================================
 * 性能优化: 标记 fuse_relu 和权重预取
 * ============================================================ */
void bnn_masknet_apply_perf_opts(bnn_masknet_t *m)
{
    if (!m || !m->g) return;

    int n_nodes = bnn_graph_node_count(m->g);

#if BNN_PERF_FUSE_BIAS_RELU
    /* 遍历图中所有 conv1d 节点, 标记可融合 ReLU 的层。
     * MaskNet 中 c1/c2/c3 (kernel>1) 后接 ReLU, head 层 (kernel=1) 后接
     * sigmoid/tanh/softplus, 不应融合。
     * 通过检查下游节点是否是 activation 来判断。
     * 简化: 所有 conv1d 都尝试融合, head 层的 sigmoid/tanh/softplus 会在
     * activation 层单独处理, 融合的 ReLU 只在 fuse_relu=1 时生效,
     * 而 head 层不会被标记 (因为下游不是 ReLU)。
     * 这里保守处理: 只标记前 3 个 conv1d (c1/c2/c3), 跳过 head 层。 */
    int fused = 0;
    int conv1d_count = 0;
    for (int i = 0; i < n_nodes; ++i) {
        bnn_layer_t *layer = bnn_graph_get_node_layer(m->g, i);
        if (!layer || !layer->type_name) continue;
        if (strcmp(layer->type_name, "conv1d") != 0) continue;
        conv1d_count++;
        /* 前 3 个 conv1d 是 c1/c2/c3 (后接 ReLU), 后面是 head 层 (后接 sigmoid/tanh/softplus) */
        if (conv1d_count <= 3) {
            if (conv1d_set_fuse_relu(layer, 1) == 0)
                fused++;
        }
    }
    BNN_LOGI("masknet perf: fuse_bias_relu=%d 层已标记 (共 %d 个 conv1d)",
             fused, conv1d_count);
#endif

#if BNN_PERF_WEIGHT_PREFETCH
    /* 预取所有 conv1d 层权重到 SRAM */
    int prefetched = 0;
    for (int i = 0; i < n_nodes; ++i) {
        bnn_layer_t *layer = bnn_graph_get_node_layer(m->g, i);
        if (!layer || !layer->type_name) continue;
        if (strcmp(layer->type_name, "conv1d") != 0) continue;
        if (conv1d_prefetch_weight_sram(layer) == 0)
            prefetched++;
    }
    BNN_LOGI("masknet perf: weight_prefetch=%d 层已预取到 SRAM", prefetched);
#endif

    BNN_PERF_PRINT_CONFIG();
}
