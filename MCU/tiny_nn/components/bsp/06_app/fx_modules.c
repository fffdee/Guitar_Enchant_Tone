/* 电吉他效果器模块实现。
 *
 * 所有效果器以 in-place 方式处理 float 缓冲 (samples[0..n-1], 范围 [-1,1])。
 * 私有状态分配在 PSRAM (延迟/混响需要大缓冲), 节点本身在内部 RAM。
 *
 * 实现要点:
 *   - 失真: tanh 软削波 / 硬削波 / 法兹 (对称削波 + 高通)
 *   - EQ:   一阶低/高架 + 一阶带通 (峰值), 计算量低
 *   - 延迟: 环形缓冲 (PSRAM), 最大 500ms
 *   - 混响: Schroeder 拓扑 (4 comb + 2 allpass), 简单但有效
 *   - 压缩: 标准 feed-forward 压缩器 (RMS 检测 + 平滑包络)
 *   - 噪声门: 简单门限检测 + 攻击/释放包络
 */
#include "fx_modules.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "fx_mod";

/* ---------- 通用工具 ---------- */
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float db_to_lin(float db) { return powf(10.0f, db / 20.0f); }
static inline float lin_to_db(float lin) { return 20.0f * log10f(lin + 1e-12f); }

/* 设置参数表 (按顺序填入 params[]) */
static void setup_params(fx_node_t *node, const fx_param_t *src, int n) {
    if (n > FX_MAX_PARAMS) n = FX_MAX_PARAMS;
    memcpy(node->params, src, (size_t)n * sizeof(fx_param_t));
    node->num_params = n;
}

/* ============================================================
 * 失真 Distortion
 * ============================================================ */
typedef struct {
    float tone_z;   /* tone 滤波器状态 (一阶低通) */
} dist_state_t;

/* 参数索引 */
enum { DIST_DRIVE, DIST_LEVEL, DIST_TONE, DIST_MODE };

static void dist_process(fx_node_t *node, float *s, int n) {
    dist_state_t *st = (dist_state_t *)node->state;
    float drive  = node->params[DIST_DRIVE].value;   /* 0~1 */
    float level  = node->params[DIST_LEVEL].value;   /* 0~1 */
    float tone   = node->params[DIST_TONE].value;    /* 0~1 (1=亮, 0=暗) */
    int   mode   = (int)node->params[DIST_MODE].value; /* 0=soft 1=hard 2=fuzz */
    /* 驱动增益: drive=0 -> 1x, drive=1 -> ~50x */
    float gain = 1.0f + drive * 49.0f;
    /* tone 一阶低通系数: tone=1 -> 不滤, tone=0 -> 重滤 */
    float alpha = 0.5f + tone * 0.49f;  /* 0.5~0.99 */
    float z = st->tone_z;

    for (int i = 0; i < n; i++) {
        float x = s[i] * gain;
        float y;
        switch (mode) {
        case 1: /* 硬削波 */
            y = clampf(x, -1.0f, 1.0f);
            break;
        case 2: /* 法兹: 对称削波 + 翻转 (方波化) */
            y = (x > 0) ? 1.0f : -1.0f;
            break;
        default: /* 软削波 tanh */
            y = tanhf(x);
            break;
        }
        /* tone 滤波 (低通) */
        z = z + alpha * (y - z);
        y = z;
        /* 输出电平 */
        s[i] = y * level;
    }
    st->tone_z = z;
}

static void dist_reset(fx_node_t *node) {
    dist_state_t *st = (dist_state_t *)node->state;
    if (st) st->tone_z = 0;
}

static void dist_deinit(fx_node_t *node) {
    if (node->state) { free(node->state); node->state = NULL; }
}

static fx_node_t *create_distortion(int sr) {
    (void)sr;
    fx_node_t *node = (fx_node_t *)calloc(1, sizeof(fx_node_t));
    if (!node) return NULL;
    dist_state_t *st = (dist_state_t *)calloc(1, sizeof(dist_state_t));
    if (!st) { free(node); return NULL; }
    node->type = FX_DISTORTION;
    node->enabled = 1;
    node->sample_rate = sr;
    node->state = st;
    node->process = dist_process;
    node->reset = dist_reset;
    node->deinit = dist_deinit;
    static const fx_param_t params[] = {
        {"drive", 0.5f, 0.0f, 1.0f, 0.5f},
        {"level", 0.7f, 0.0f, 1.0f, 0.7f},
        {"tone",  0.7f, 0.0f, 1.0f, 0.7f},
        {"mode",  0.0f, 0.0f, 2.0f, 0.0f},  /* 0=soft 1=hard 2=fuzz */
    };
    setup_params(node, params, 4);
    return node;
}

/* ============================================================
 * 三段均衡 EQ (低架 + 峰值 + 高架, 一阶)
 * ============================================================ */
typedef struct {
    float low_z1, mid_z1, mid_z2, high_z1;
} eq_state_t;

enum { EQ_LOW, EQ_MID, EQ_HIGH, EQ_LOW_FREQ, EQ_HIGH_FREQ };

static void eq_process(fx_node_t *node, float *s, int n) {
    eq_state_t *st = (eq_state_t *)node->state;
    float lg = db_to_lin(node->params[EQ_LOW].value);
    float mg = db_to_lin(node->params[EQ_MID].value);
    float hg = db_to_lin(node->params[EQ_HIGH].value);
    int sr = node->sample_rate;
    /* 一阶低架: 截止 low_freq 以下增益 lg */
    float lf = node->params[EQ_LOW_FREQ].value;
    float hf = node->params[EQ_HIGH_FREQ].value;
    /* 系数 (简化一阶 shelving) */
    float wl = 2.0f * 3.14159265f * lf / sr;
    float al = sinf(wl) / (sinf(wl) + 1.0f);   /* 低架系数 */
    float wh = 2.0f * 3.14159265f * hf / sr;
    float ah = sinf(wh) / (sinf(wh) + 1.0f);   /* 高架系数 */
    /* 中频峰值: 中心 1kHz, 一阶带通近似 */
    float wm = 2.0f * 3.14159265f * 1000.0f / sr;
    float am = sinf(wm) / (sinf(wm) + 1.0f);

    float lz = st->low_z1, mz1 = st->mid_z1, mz2 = st->mid_z2, hz = st->high_z1;
    for (int i = 0; i < n; i++) {
        float x = s[i];
        /* 低架: 低频部分乘 lg */
        lz = lz + al * (x - lz);
        float low = lz;
        /* 高架: 高频部分乘 hg */
        hz = hz + ah * (x - hz);
        float high = hz;
        /* 中频带通 */
        mz1 = mz1 + am * (x - mz1);
        mz2 = mz2 + am * (mz1 - mz2);
        float mid = mz2;
        /* 合成: 原始 - (低+高) + 低*lg + 高*hg + 中*(mg-1) */
        s[i] = x + low * (lg - 1.0f) + high * (hg - 1.0f) + mid * (mg - 1.0f);
    }
    st->low_z1 = lz; st->mid_z1 = mz1; st->mid_z2 = mz2; st->high_z1 = hz;
}

static void eq_reset(fx_node_t *node) {
    eq_state_t *st = (eq_state_t *)node->state;
    if (st) { st->low_z1 = st->mid_z1 = st->mid_z2 = st->high_z1 = 0; }
}

static void eq_deinit(fx_node_t *node) {
    if (node->state) { free(node->state); node->state = NULL; }
}

static fx_node_t *create_eq(int sr) {
    fx_node_t *node = (fx_node_t *)calloc(1, sizeof(fx_node_t));
    if (!node) return NULL;
    eq_state_t *st = (eq_state_t *)calloc(1, sizeof(eq_state_t));
    if (!st) { free(node); return NULL; }
    node->type = FX_EQ;
    node->enabled = 1;
    node->sample_rate = sr;
    node->state = st;
    node->process = eq_process;
    node->reset = eq_reset;
    node->deinit = eq_deinit;
    static const fx_param_t params[] = {
        {"low_db",   0.0f, -12.0f, 12.0f, 0.0f},
        {"mid_db",   0.0f, -12.0f, 12.0f, 0.0f},
        {"high_db",  0.0f, -12.0f, 12.0f, 0.0f},
        {"low_freq", 200.0f, 50.0f, 1000.0f, 200.0f},
        {"high_freq",3000.0f,1000.0f,8000.0f,3000.0f},
    };
    setup_params(node, params, 5);
    return node;
}

/* ============================================================
 * 延迟 Delay (环形缓冲, PSRAM)
 * ============================================================ */
typedef struct {
    float *buf;       /* 延迟缓冲 (PSRAM) */
    int    buf_size;  /* 缓冲长度 (采样数) */
    int    write_idx;
} delay_state_t;

enum { DEL_TIME_MS, DEL_FEEDBACK, DEL_MIX, DEL_LEVEL };

static void delay_process(fx_node_t *node, float *s, int n) {
    delay_state_t *st = (delay_state_t *)node->state;
    if (!st || !st->buf) return;
    float time_ms = node->params[DEL_TIME_MS].value;
    float fb      = node->params[DEL_FEEDBACK].value;
    float mix     = node->params[DEL_MIX].value;
    float level   = node->params[DEL_LEVEL].value;
    int sr = node->sample_rate;
    int delay_samples = (int)(time_ms * sr / 1000.0f);
    if (delay_samples < 1) delay_samples = 1;
    if (delay_samples >= st->buf_size) delay_samples = st->buf_size - 1;

    int w = st->write_idx;
    for (int i = 0; i < n; i++) {
        /* 读位置 = 写位置 - 延迟 */
        int r = w - delay_samples;
        if (r < 0) r += st->buf_size;
        float delayed = st->buf[r];
        /* 写入: 当前输入 + 反馈 */
        st->buf[w] = s[i] + fb * delayed;
        /* 输出: 干信号 * (1-mix) + 湿信号 * mix, 再乘 level */
        s[i] = (s[i] * (1.0f - mix) + delayed * mix) * level;
        w++;
        if (w >= st->buf_size) w = 0;
    }
    st->write_idx = w;
}

static void delay_reset(fx_node_t *node) {
    delay_state_t *st = (delay_state_t *)node->state;
    if (st && st->buf) { memset(st->buf, 0, sizeof(float) * st->buf_size); st->write_idx = 0; }
}

static void delay_deinit(fx_node_t *node) {
    delay_state_t *st = (delay_state_t *)node->state;
    if (st) {
        if (st->buf) heap_caps_free(st->buf);
        free(st);
    }
    node->state = NULL;
}

static fx_node_t *create_delay(int sr) {
    fx_node_t *node = (fx_node_t *)calloc(1, sizeof(fx_node_t));
    if (!node) return NULL;
    delay_state_t *st = (delay_state_t *)calloc(1, sizeof(delay_state_t));
    if (!st) { free(node); return NULL; }
    /* 最大 500ms 缓冲, 分配在 PSRAM */
    int max_samples = sr / 2;  /* 500ms */
    st->buf = (float *)heap_caps_malloc((size_t)max_samples * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!st->buf) {
        ESP_LOGW(TAG, "PSRAM 分配失败, 延迟缓冲降级到内部 RAM");
        st->buf = (float *)heap_caps_malloc((size_t)max_samples * sizeof(float), MALLOC_CAP_INTERNAL);
        if (!st->buf) { free(st); free(node); return NULL; }
    }
    st->buf_size = max_samples;
    st->write_idx = 0;
    memset(st->buf, 0, sizeof(float) * max_samples);

    node->type = FX_DELAY;
    node->enabled = 1;
    node->sample_rate = sr;
    node->state = st;
    node->process = delay_process;
    node->reset = delay_reset;
    node->deinit = delay_deinit;
    static const fx_param_t params[] = {
        {"time_ms",  250.0f, 10.0f, 500.0f, 250.0f},
        {"feedback", 0.4f,   0.0f,  0.9f,   0.4f},
        {"mix",      0.3f,   0.0f,  1.0f,   0.3f},
        {"level",    1.0f,   0.0f,  1.0f,   1.0f},
    };
    setup_params(node, params, 4);
    return node;
}

/* ============================================================
 * 混响 Reverb (Schroeder: 4 comb + 2 allpass)
 * ============================================================ */
#define REVERB_NCOMB   4
#define REVERB_NALL    2

typedef struct {
    float *comb_buf[REVERB_NCOMB];
    int    comb_size[REVERB_NCOMB];
    int    comb_idx[REVERB_NCOMB];
    float  comb_lp[REVERB_NCOMB];   /* 每个 comb 的阻尼低通状态 */
    float *all_buf[REVERB_NALL];
    int    all_size[REVERB_NALL];
    int    all_idx[REVERB_NALL];
} reverb_state_t;

enum { REV_ROOM, REV_DAMP, REV_MIX, REV_LEVEL };

static void reverb_process(fx_node_t *node, float *s, int n) {
    reverb_state_t *st = (reverb_state_t *)node->state;
    if (!st) return;
    float room = node->params[REV_ROOM].value;   /* 0~1 房间大小 (反馈系数) */
    float damp = node->params[REV_DAMP].value;   /* 0~1 阻尼 (低通) */
    float mix  = node->params[REV_MIX].value;
    float level = node->params[REV_LEVEL].value;
    /* 反馈系数: room=0 -> 0.3, room=1 -> 0.9 */
    float fb = 0.3f + room * 0.6f;
    /* 阻尼低通系数 */
    float damp_a = 1.0f - damp * 0.9f;

    /* allpass 固定反馈系数 */
    const float all_g = 0.7f;

    for (int i = 0; i < n; i++) {
        float x = s[i];
        float wet = 0;
        /* comb 滤波器: y = x + fb * lp */
        for (int k = 0; k < REVERB_NCOMB; k++) {
            float delayed = st->comb_buf[k][st->comb_idx[k]];
            /* 阻尼低通 */
            st->comb_lp[k] = st->comb_lp[k] + damp_a * (delayed - st->comb_lp[k]);
            float out = x + fb * st->comb_lp[k];
            st->comb_buf[k][st->comb_idx[k]] = out;
            if (++st->comb_idx[k] >= st->comb_size[k]) st->comb_idx[k] = 0;
            wet += st->comb_lp[k];
        }
        wet /= REVERB_NCOMB;
        /* allpass 串联 */
        for (int k = 0; k < REVERB_NALL; k++) {
            float delayed = st->all_buf[k][st->all_idx[k]];
            float out = -all_g * x + delayed;
            st->all_buf[k][st->all_idx[k]] = x + all_g * delayed;
            if (++st->all_idx[k] >= st->all_size[k]) st->all_idx[k] = 0;
            x = out;  /* 下一级输入 */
        }
        wet = x;
        /* 干湿混合 */
        s[i] = (s[i] * (1.0f - mix) + wet * mix) * level;
    }
}

static void reverb_reset(fx_node_t *node) {
    reverb_state_t *st = (reverb_state_t *)node->state;
    if (!st) return;
    for (int k = 0; k < REVERB_NCOMB; k++) {
        if (st->comb_buf[k]) memset(st->comb_buf[k], 0, sizeof(float) * st->comb_size[k]);
        st->comb_idx[k] = 0;
        st->comb_lp[k] = 0;
    }
    for (int k = 0; k < REVERB_NALL; k++) {
        if (st->all_buf[k]) memset(st->all_buf[k], 0, sizeof(float) * st->all_size[k]);
        st->all_idx[k] = 0;
    }
}

static void reverb_deinit(fx_node_t *node) {
    reverb_state_t *st = (reverb_state_t *)node->state;
    if (st) {
        for (int k = 0; k < REVERB_NCOMB; k++)
            if (st->comb_buf[k]) heap_caps_free(st->comb_buf[k]);
        for (int k = 0; k < REVERB_NALL; k++)
            if (st->all_buf[k]) heap_caps_free(st->all_buf[k]);
        free(st);
    }
    node->state = NULL;
}

static fx_node_t *create_reverb(int sr) {
    fx_node_t *node = (fx_node_t *)calloc(1, sizeof(fx_node_t));
    if (!node) return NULL;
    reverb_state_t *st = (reverb_state_t *)calloc(1, sizeof(reverb_state_t));
    if (!st) { free(node); return NULL; }
    /* Schroeder 经典延迟长度 (按 48kHz 比例缩放) */
    /* comb: 1116, 1188, 1277, 1356 (44.1kHz) -> 48kHz 比例 1.088 */
    float scale = (float)sr / 44100.0f;
    int comb_lens[REVERB_NCOMB] = {
        (int)(1116 * scale), (int)(1188 * scale),
        (int)(1277 * scale), (int)(1356 * scale)
    };
    int all_lens[REVERB_NALL] = {
        (int)(556 * scale), (int)(441 * scale)
    };
    int ok = 1;
    for (int k = 0; k < REVERB_NCOMB && ok; k++) {
        st->comb_size[k] = comb_lens[k];
        st->comb_buf[k] = (float *)heap_caps_malloc(sizeof(float) * comb_lens[k], MALLOC_CAP_SPIRAM);
        if (!st->comb_buf[k]) ok = 0;
        else memset(st->comb_buf[k], 0, sizeof(float) * comb_lens[k]);
    }
    for (int k = 0; k < REVERB_NALL && ok; k++) {
        st->all_size[k] = all_lens[k];
        st->all_buf[k] = (float *)heap_caps_malloc(sizeof(float) * all_lens[k], MALLOC_CAP_SPIRAM);
        if (!st->all_buf[k]) ok = 0;
        else memset(st->all_buf[k], 0, sizeof(float) * all_lens[k]);
    }
    if (!ok) {
        ESP_LOGE(TAG, "混响缓冲分配失败");
        reverb_deinit(node);
        free(node);
        return NULL;
    }
    node->type = FX_REVERB;
    node->enabled = 1;
    node->sample_rate = sr;
    node->state = st;
    node->process = reverb_process;
    node->reset = reverb_reset;
    node->deinit = reverb_deinit;
    static const fx_param_t params[] = {
        {"room",  0.5f, 0.0f, 1.0f, 0.5f},
        {"damp",  0.3f, 0.0f, 1.0f, 0.3f},
        {"mix",   0.3f, 0.0f, 1.0f, 0.3f},
        {"level", 1.0f, 0.0f, 1.0f, 1.0f},
    };
    setup_params(node, params, 4);
    return node;
}

/* ============================================================
 * 压缩器 Compressor (feed-forward, RMS 检测)
 * ============================================================ */
typedef struct {
    float env;       /* 包络 (RMS 平滑) */
} comp_state_t;

enum { COMP_THRESHOLD, COMP_RATIO, COMP_ATTACK, COMP_RELEASE, COMP_MAKEUP };

static void comp_process(fx_node_t *node, float *s, int n) {
    comp_state_t *st = (comp_state_t *)node->state;
    if (!st) return;
    float thr_db = node->params[COMP_THRESHOLD].value;
    float ratio  = node->params[COMP_RATIO].value;
    float atk_ms = node->params[COMP_ATTACK].value;
    float rel_ms = node->params[COMP_RELEASE].value;
    float makeup = node->params[COMP_MAKEUP].value;  /* dB */
    int sr = node->sample_rate;
    /* 攻击/释放系数 (一阶平滑) */
    float atk_g = atk_ms > 0 ? expf(-1.0f / (sr * atk_ms / 1000.0f)) : 0;
    float rel_g = rel_ms > 0 ? expf(-1.0f / (sr * rel_ms / 1000.0f)) : 0;
    float makeup_lin = db_to_lin(makeup);
    float env = st->env;

    for (int i = 0; i < n; i++) {
        float x = s[i];
        float abs_x = fabsf(x);
        /* 平方检测 -> RMS 近似 */
        float det = abs_x;  /* 简化用绝对值 */
        /* 包络跟随 */
        float g;
        if (det > env) g = atk_g;  /* 攻击 */
        else           g = rel_g;  /* 释放 */
        env = g * env + (1.0f - g) * det;
        /* 增益计算 */
        float env_db = lin_to_db(env);
        float gain_db = 0;
        if (env_db > thr_db) {
            float over = env_db - thr_db;
            gain_db = -over * (1.0f - 1.0f / ratio);
        }
        float gain_lin = db_to_lin(gain_db);
        s[i] = x * gain_lin * makeup_lin;
    }
    st->env = env;
}

static void comp_reset(fx_node_t *node) {
    comp_state_t *st = (comp_state_t *)node->state;
    if (st) st->env = 0;
}

static void comp_deinit(fx_node_t *node) {
    if (node->state) { free(node->state); node->state = NULL; }
}

static fx_node_t *create_compressor(int sr) {
    fx_node_t *node = (fx_node_t *)calloc(1, sizeof(fx_node_t));
    if (!node) return NULL;
    comp_state_t *st = (comp_state_t *)calloc(1, sizeof(comp_state_t));
    if (!st) { free(node); return NULL; }
    node->type = FX_COMPRESSOR;
    node->enabled = 1;
    node->sample_rate = sr;
    node->state = st;
    node->process = comp_process;
    node->reset = comp_reset;
    node->deinit = comp_deinit;
    static const fx_param_t params[] = {
        {"threshold", -24.0f, -60.0f, 0.0f, -24.0f},  /* dB */
        {"ratio",      4.0f,   1.0f,  20.0f, 4.0f},
        {"attack",     5.0f,   0.1f,  100.0f, 5.0f},  /* ms */
        {"release",   50.0f,   10.0f, 1000.0f, 50.0f}, /* ms */
        {"makeup",     0.0f,   0.0f,  24.0f, 0.0f},   /* dB */
    };
    setup_params(node, params, 5);
    return node;
}

/* ============================================================
 * 噪声门 Noise Gate
 * ============================================================ */
typedef struct {
    float env;       /* 包络 */
    float gain;      /* 当前增益 (0~1) */
} gate_state_t;

enum { GATE_THRESHOLD, GATE_ATTACK, GATE_RELEASE, GATE_HOLD };

static void gate_process(fx_node_t *node, float *s, int n) {
    gate_state_t *st = (gate_state_t *)node->state;
    if (!st) return;
    float thr_db = node->params[GATE_THRESHOLD].value;
    float atk_ms = node->params[GATE_ATTACK].value;
    float rel_ms = node->params[GATE_RELEASE].value;
    int sr = node->sample_rate;
    float thr_lin = db_to_lin(thr_db);
    float atk_g = atk_ms > 0 ? expf(-1.0f / (sr * atk_ms / 1000.0f)) : 0;
    float rel_g = rel_ms > 0 ? expf(-1.0f / (sr * rel_ms / 1000.0f)) : 0;
    float env = st->env;
    float gain = st->gain;
    float target;

    for (int i = 0; i < n; i++) {
        float x = s[i];
        float det = fabsf(x);
        env = env + 0.01f * (det - env);  /* 简单包络 */
        target = (env > thr_lin) ? 1.0f : 0.0f;
        /* 平滑增益 */
        float g = (target > gain) ? atk_g : rel_g;
        gain = g * gain + (1.0f - g) * target;
        s[i] = x * gain;
    }
    st->env = env;
    st->gain = gain;
}

static void gate_reset(fx_node_t *node) {
    gate_state_t *st = (gate_state_t *)node->state;
    if (st) { st->env = 0; st->gain = 0; }
}

static void gate_deinit(fx_node_t *node) {
    if (node->state) { free(node->state); node->state = NULL; }
}

static fx_node_t *create_noise_gate(int sr) {
    fx_node_t *node = (fx_node_t *)calloc(1, sizeof(fx_node_t));
    if (!node) return NULL;
    gate_state_t *st = (gate_state_t *)calloc(1, sizeof(gate_state_t));
    if (!st) { free(node); return NULL; }
    node->type = FX_NOISE_GATE;
    node->enabled = 1;
    node->sample_rate = sr;
    node->state = st;
    node->process = gate_process;
    node->reset = gate_reset;
    node->deinit = gate_deinit;
    static const fx_param_t params[] = {
        {"threshold", -50.0f, -80.0f, -20.0f, -50.0f},  /* dB */
        {"attack",     1.0f,   0.1f,  50.0f,  1.0f},    /* ms */
        {"release",  100.0f,   10.0f, 500.0f, 100.0f},  /* ms */
        {"hold",      10.0f,   0.0f,  500.0f, 10.0f},   /* ms (简化未用) */
    };
    setup_params(node, params, 4);
    return node;
}

/* ============================================================
 * 公共 API
 * ============================================================ */
fx_node_t *fx_node_create(fx_type_t type, int sample_rate) {
    switch (type) {
    case FX_DISTORTION: return create_distortion(sample_rate);
    case FX_EQ:         return create_eq(sample_rate);
    case FX_DELAY:      return create_delay(sample_rate);
    case FX_REVERB:     return create_reverb(sample_rate);
    case FX_COMPRESSOR: return create_compressor(sample_rate);
    case FX_NOISE_GATE: return create_noise_gate(sample_rate);
    default:
        ESP_LOGW(TAG, "未知效果器类型: %d", (int)type);
        return NULL;
    }
}

void fx_node_destroy(fx_node_t *node) {
    if (!node) return;
    if (node->deinit) node->deinit(node);
    free(node);
}

int fx_node_param_index(const fx_node_t *node, const char *name) {
    if (!node || !name) return -1;
    for (int i = 0; i < node->num_params; i++) {
        if (!strcmp(node->params[i].name, name)) return i;
    }
    return -1;
}

int fx_node_set_param(fx_node_t *node, int idx, float value) {
    if (!node || idx < 0 || idx >= node->num_params) return -1;
    node->params[idx].value = clampf(value, node->params[idx].min_val, node->params[idx].max_val);
    return 0;
}

int fx_node_set_param_by_name(fx_node_t *node, const char *name, float value) {
    int idx = fx_node_param_index(node, name);
    if (idx < 0) return -1;
    return fx_node_set_param(node, idx, value);
}

static const char *s_type_names[] = {
    "none", "distortion", "eq", "delay", "reverb", "compressor", "noise_gate"
};

const char *fx_type_name(fx_type_t type) {
    if (type <= FX_NONE || type >= FX_TYPE_MAX) return "none";
    return s_type_names[type];
}

fx_type_t fx_type_from_name(const char *name) {
    if (!name) return FX_NONE;
    for (int i = 1; i < FX_TYPE_MAX; i++) {
        if (!strcmp(name, s_type_names[i])) return (fx_type_t)i;
    }
    return FX_NONE;
}

void fx_list_types(char *buf, size_t n) {
    if (!buf || n == 0) return;
    buf[0] = 0;
    for (int i = 1; i < FX_TYPE_MAX; i++) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%s%s", (i > 1) ? " | " : "", s_type_names[i]);
        strncat(buf, tmp, n - strlen(buf) - 1);
    }
}
