#include "audio_xform.h"
#include "model_store.h"
#include "wav.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "bnn_model/bnn_masknet.h"
#include "bnn_frontend/bnn_mask_config.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include "bnn_utils/bnn_workspace.h"
#include "bnn_op/bnn_dsp.h"
#include "bnn_op/bnn_op.h"
#include "bnn_layer/bnn_xform_layers.h"
#include "dsp_accel.h"
#include "nn_accel.h"
#include "dspm_mult.h"

static const char *TAG = "xform";
#define BLOCK_FRAMES 64

/* 含 INT8 约 740880B; 无 INT8 约 598832B */
#define XFORM_MODEL_EXPECT_BYTES 740880u
#define XFORM_MODEL_MIN_BYTES    590000u
#define XFORM_MODEL_MAX_BYTES    750000u

static void validate_model_pkg(const model_pkg_t *pkg)
{
    if (!pkg) return;
    ESP_LOGI(TAG, "model bin: %zu B (含INT8~%u / 无INT8~599KB)", pkg->buf_size, XFORM_MODEL_EXPECT_BYTES);
    if (pkg->buf_size < XFORM_MODEL_MIN_BYTES || pkg->buf_size > XFORM_MODEL_MAX_BYTES)
        ESP_LOGW(TAG, "bin 大小异常, 请确认 SD:/model/xform_model.bin 来自 exports/");
    if (pkg->weights_i8 && pkg->weights_i8_size > 0)
        ESP_LOGI(TAG, "bin 含 weights_i8 (%zuB), infer -8 可启用 INT8", pkg->weights_i8_size);
    else
        ESP_LOGI(TAG, "bin 无 weights_i8, 仅 F32 推理");

    if (pkg->mel_mean && pkg->mel_std && pkg->mel_n >= 8) {
        ESP_LOGI(TAG, "mel_norm[0:3]: mean=%.3f %.3f %.3f  std=%.3f %.3f %.3f",
                 pkg->mel_mean[0], pkg->mel_mean[1], pkg->mel_mean[2],
                 pkg->mel_std[0], pkg->mel_std[1], pkg->mel_std[2]);
        if (fabsf(pkg->mel_mean[0]) < 0.01f && fabsf(pkg->mel_mean[1]) < 0.01f &&
            fabsf(pkg->mel_std[0] - 1.0f) < 0.01f) {
            ESP_LOGW(TAG, "mel_mean/std 似占位值 (0,1,...); 请用 checkpoint 重导出 xform_model.bin");
        }
    }

    if (pkg->weights && pkg->weights_size > 16) {
        const float *w = (const float *)((const uint8_t *)pkg->weights + 16);
        float wabs = 0.0f;
        for (int i = 0; i < 256; ++i) wabs += fabsf(w[i]);
        ESP_LOGI(TAG, "weights sanity: w[0]=%.4f  sum|w[0:256]|=%.2f", w[0], wabs);
        if (wabs < 1e-4f)
            ESP_LOGE(TAG, "weights 似全零 — CNN 不会变换音色, 请更换 xform_model.bin");
    }
}

/* 把 tinynn 内部日志转到 ESP 串口 (MCU 默认 sink 为 no-op, 否则看不到失败原因) */
static void bnn_log_to_esp(int level, const char *msg, void *user)
{
    (void)user;
    if (level <= 1) ESP_LOGE("bnn", "%s", msg);
    else            ESP_LOGI("bnn", "%s", msg);
}

model_pkg_t    s_pkg;
bnn_masknet_t *s_net = NULL;
bnn_mask_cfg_t s_cfg;
static int            s_loaded = 0;
static int            s_int8_ready = 0;      /* weights_i8 已注入且 ESP-NN 可用 */
static void          *s_ws_sram_buf = NULL;  /* workspace 内部 SRAM 缓冲 */

/*
 * workspace 缓冲大小: 必须 ≥ 推理过程中的峰值 workspace 用量.
 * 该值由 bnn_ws_peak() 观测得到; 当前网络 (hidden=128, k=3, T=64)
 * 最大 col = 128*3*64 floats = 96KB, 预留 1.5 倍余量.
 */
#define BNN_WS_SRAM_SIZE (160 * 1024)

/* ── 推理时间统计累加器 ──────────────────────────────────────────────────── */
typedef struct {
    uint32_t n;            /* 成功推理次数      */
    int64_t  sum_ms;       /* 累计推理时间 ms   */
    int64_t  min_ms;       /* 单次最短 ms       */
    int64_t  max_ms;       /* 单次最长 ms       */
    int64_t  sum_out_smp;  /* 累计输出采样点数  */
    size_t   ws_peak_kb;   /* 历史最大 WS KB   */
} xform_stats_t;

static xform_stats_t s_stats = {
    .n           = 0,
    .sum_ms      = 0,
    .min_ms      = INT64_MAX,
    .max_ms      = 0,
    .sum_out_smp = 0,
    .ws_peak_kb  = 0,
};

static void *psram_malloc(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    return p ? p : malloc(sz);
}

/* tinynn 内存分配器: 大块(卷积权重~596KB / 展开矩阵)走 PSRAM, 小块走内部 RAM.
 * 阈值取 16KB: 让 FFT 临时缓冲(frame_tmp/spec/time 等 ~n_fft 大小)留在内部 RAM,
 * 避免 cfft/irfft 频繁随机访问 PSRAM 拖慢; 否则 596KB 权重在内部 RAM 必然 OOM.
 *
 * 16 字节对齐: ESP32-P4 PIE 128-bit 向量指令要求操作数地址 16 字节对齐.
 * heap_caps_aligned_alloc(16, ...) 保证 workspace buf 基址、权重张量均满足此要求,
 * 配合 bnn_workspace.c 中 WS_ALIGN(16) 确保 col/tmp scratch 内偏移也对齐. */
#define BNN_PSRAM_THRESHOLD (16 * 1024)
#define BNN_PIE_ALIGN       16u
static void *bnn_psram_alloc(size_t sz)
{
    /* 对齐分配大小至 BNN_PIE_ALIGN 的整倍数 (heap_caps_aligned_alloc 要求) */
    size_t aligned_sz = (sz + BNN_PIE_ALIGN - 1u) & ~(BNN_PIE_ALIGN - 1u);
    if (aligned_sz == 0) aligned_sz = BNN_PIE_ALIGN;

    uint32_t caps = (sz >= BNN_PSRAM_THRESHOLD) ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
    void *p = heap_caps_aligned_alloc(BNN_PIE_ALIGN, aligned_sz, caps);
    if (!p) p = heap_caps_aligned_alloc(BNN_PIE_ALIGN, aligned_sz, MALLOC_CAP_SPIRAM);
    if (!p) p = heap_caps_aligned_alloc(BNN_PIE_ALIGN, aligned_sz, MALLOC_CAP_DEFAULT);
    return p;
}
static void bnn_psram_free(void *p) { heap_caps_free(p); }

static void cfg_from_pkg(bnn_mask_cfg_t *cfg, const model_pkg_t *pkg)
{
    bnn_mask_cfg_default(cfg);
    const int32_t *c = pkg->config;
    if (c && pkg->config_n >= 14) {
        cfg->sample_rate = c[1]; cfg->n_fft = c[2]; cfg->hop = c[3]; cfg->n_mels = c[4];
        cfg->fmin = (float)c[5]; cfg->fmax = (float)c[6]; cfg->gmax = (float)c[7] / 1000.0f;
        cfg->phase_bands = c[8]; cfg->noise_bands = c[9]; cfg->hidden = c[10];
        cfg->kernel = c[11]; cfg->dilation2 = c[12]; cfg->emb_dim = c[13];
    }
    cfg->n_bins = cfg->n_fft / 2 + 1;
}

esp_err_t audio_xform_init(const char *model_path)
{
    if (s_loaded) return ESP_OK;

    /* tinynn 日志转串口 + 内部分配走 PSRAM, 必须在 create 之前设置 */
    bnn_log_set_sink(bnn_log_to_esp, NULL);
    bnn_mem_set_allocator(bnn_psram_alloc, bnn_psram_free);

    /* GEMM 微基准 (在 workspace 160KB 占用前运行, 避免内部 SRAM 不足) */
    {
        const int BM = 128, BK = 384, BN = 64;
        const int BENCH_ITER = 20;
        float *bA_psram = heap_caps_aligned_alloc(BNN_PIE_ALIGN, (size_t)BM * BK * 4,
                                                  MALLOC_CAP_SPIRAM);
        float *bA_sram  = heap_caps_aligned_alloc(BNN_PIE_ALIGN, (size_t)BM * BK * 4,
                                                  MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
        float *bB_sram  = heap_caps_aligned_alloc(BNN_PIE_ALIGN, (size_t)BK * BN * 4,
                                                  MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
        float *bC_sram  = heap_caps_aligned_alloc(BNN_PIE_ALIGN, (size_t)BM * BN * 4,
                                                  MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
        if (bA_psram && bB_sram && bC_sram) {
            memset(bA_psram, 0, (size_t)BM * BK * 4);
            if (bA_sram) memset(bA_sram, 0, (size_t)BM * BK * 4);
            memset(bB_sram, 0, (size_t)BK * BN * 4);
            memset(bC_sram, 0, (size_t)BM * BN * 4);
            dspm_mult_f32(bA_psram, bB_sram, bC_sram, BM, BK, BN);
            int64_t t0 = esp_timer_get_time();
            for (int _i = 0; _i < BENCH_ITER; ++_i)
                dspm_mult_f32(bA_psram, bB_sram, bC_sram, BM, BK, BN);
            int64_t us_psram = (esp_timer_get_time() - t0) / BENCH_ITER;
            int64_t us_sram = -1;
            if (bA_sram) {
                dspm_mult_f32(bA_sram, bB_sram, bC_sram, BM, BK, BN);
                t0 = esp_timer_get_time();
                for (int _i = 0; _i < BENCH_ITER; ++_i)
                    dspm_mult_f32(bA_sram, bB_sram, bC_sram, BM, BK, BN);
                us_sram = (esp_timer_get_time() - t0) / BENCH_ITER;
            }
            if (us_sram >= 0)
                ESP_LOGI(TAG, "[bench] GEMM %dx%dx%d | A=PSRAM: %lldus  A=SRAM: %lldus",
                         BM, BK, BN, us_psram, us_sram);
            else
                ESP_LOGI(TAG, "[bench] GEMM %dx%dx%d | A=PSRAM: %lldus  (A=SRAM OOM)",
                         BM, BK, BN, us_psram);
        } else {
            ESP_LOGW(TAG, "[bench] GEMM 基准内存分配失败, 跳过");
        }
        if (bA_psram) heap_caps_free(bA_psram);
        if (bA_sram)  heap_caps_free(bA_sram);
        if (bB_sram)  heap_caps_free(bB_sram);
        if (bC_sram)  heap_caps_free(bC_sram);
    }

    /* workspace 固定到内部 SRAM: conv1d col/tmp scratch 全程在快速 SRAM,
     * 避免 GEMM 反复穿越慢速 PSRAM 总线 (不做此步推理速度约 16x 于实时). */
    if (!s_ws_sram_buf) {
        s_ws_sram_buf = heap_caps_aligned_alloc(
                BNN_PIE_ALIGN, BNN_WS_SRAM_SIZE,
                MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    }
    if (s_ws_sram_buf) {
        bnn_ws_assign_buf(bnn_workspace_default(), s_ws_sram_buf, BNN_WS_SRAM_SIZE);
        ESP_LOGI(TAG, "workspace: %dKB 内部 SRAM (col/tmp scratch 在 SRAM)",
                 BNN_WS_SRAM_SIZE / 1024);
    } else {
        ESP_LOGW(TAG, "workspace SRAM 分配失败 (%dKB), 降级 PSRAM (推理会很慢!)",
                 BNN_WS_SRAM_SIZE / 1024);
    }

    esp_err_t err = model_store_load(model_path, &s_pkg);
    if (err != ESP_OK) return err;
    validate_model_pkg(&s_pkg);

    cfg_from_pkg(&s_cfg, &s_pkg);
    int num_inst = s_pkg.num_inst > 0 ? s_pkg.num_inst : 1;

    /* 接入 ESP32-P4 esp-dsp 单精度 FFT 后端 (替换默认 double-twiddle 软件实现) */
    if (dsp_accel_init(s_cfg.n_fft) == ESP_OK) {
        bnn_dsp_set_backend(dsp_accel_backend());
        ESP_LOGI(TAG, "FFT 后端: esp-dsp P4 (n_fft=%d)", s_cfg.n_fft);
    } else {
        ESP_LOGW(TAG, "esp-dsp 初始化失败, 回退默认 FFT (较慢)");
    }

    /* 接入 ESP32-P4 PIE 向量 NN 算子后端 (替换 cpu_ref GEMM/vec 慢路径).
     * Phase-1: dspm_mult_f32_arp4 (128-bit PIE GEMM) + dsps_dotprod/add/mul.
     * Phase-2: nn_accel_init_int8() 启用 INT8 ESP-NN conv (当前默认禁用, 见下方). */
    if (nn_accel_init() == ESP_OK) {
        bnn_op_set_backend(nn_accel_backend());
        ESP_LOGI(TAG, "NN op 后端: esp-dsp P4 PIE GEMM/vec (Phase-1)");
    } else {
        ESP_LOGW(TAG, "nn_accel 初始化失败, 保留 cpu_ref 后端");
    }

    /* 有图 IR 段则数据驱动建图 (改网络=换模型文件), 否则回退内置硬编码图 */
    if (s_pkg.graph_ir && s_pkg.graph_ir_size > 0)
        ESP_LOGI(TAG, "模型含图 IR 段 (%zuB), 数据驱动建图", s_pkg.graph_ir_size);
    s_net = bnn_masknet_create_ir(&s_cfg, num_inst, BLOCK_FRAMES,
                                  s_pkg.graph_ir, s_pkg.graph_ir_size);
    if (!s_net) {
        ESP_LOGE(TAG, "masknet_create failed (OOM? need PSRAM)");
        model_store_free(&s_pkg);
        return ESP_ERR_NO_MEM;
    }
    if (bnn_masknet_load_weights_mem(s_net, s_pkg.weights, s_pkg.weights_size) != 0) {
        ESP_LOGE(TAG, "load_weights_mem failed (param mismatch?)");
        bnn_masknet_destroy(s_net); s_net = NULL; model_store_free(&s_pkg);
        return ESP_FAIL;
    }
    bnn_masknet_set_mel_norm(s_net, s_pkg.mel_mean, s_pkg.mel_std);
    bnn_masknet_set_embedding_table(s_net, s_pkg.emb, num_inst, s_pkg.emb_dim);

    /* Phase 2 INT8: 预注入权重; 默认 F32 推理, infer/live -8 启用运行时开关. */
    s_int8_ready = 0;
    bnn_conv1d_set_int8_runtime(0);
    if (s_pkg.weights_i8 && s_pkg.weights_i8_size > 0) {
        int inj = bnn_masknet_load_weights_i8_mem(s_net,
                                                  s_pkg.weights_i8,
                                                  s_pkg.weights_i8_size);
        if (inj > 0 && nn_accel_init_int8() == ESP_OK) {
            s_int8_ready = 1;
            ESP_LOGI(TAG, "INT8 就绪: %d conv 层已注入 (默认 F32, infer -8 启用)", inj);
        } else {
            ESP_LOGW(TAG, "INT8 注入/初始化失败 (inj=%d), 仅 F32", inj);
        }
    } else {
        ESP_LOGI(TAG, "模型不含 INT8 段, 仅 F32 推理");
    }

    /* 注入 esp_timer 计时函数用于推理分段诊断 */
    bnn_masknet_set_tick_fn(s_net, esp_timer_get_time);

    s_loaded = 1;
    ESP_LOGI(TAG, "ready: %d instruments, sr=%d params=%zu",
             num_inst, s_cfg.sample_rate, bnn_masknet_num_params(s_net));
    return ESP_OK;
}

esp_err_t audio_xform_file(const char *in_wav, const char *out_wav,
                           const char *instrument, const audio_xform_opt_t *opt)
{
    if (!s_loaded || !s_net) { ESP_LOGE(TAG, "not initialized"); return ESP_ERR_INVALID_STATE; }

    int id = model_store_instrument_id(&s_pkg, instrument);
    if (id < 0) {
        /* 模型不含该乐器: 跳过, 不回退到其它乐器 */
        ESP_LOGW(TAG, "instrument '%s' 不在模型中, 跳过", instrument);
        return ESP_ERR_NOT_FOUND;
    }
    bnn_masknet_set_instrument(s_net, id);
    /* 噪声注入: 默认关闭, opt->add_noise=1 时开启 */
    bnn_masknet_set_add_noise(s_net, opt ? opt->add_noise : 0);

    float *x = NULL; int n = 0, sr = 0;
    esp_err_t err = wav_read_mono_f32(in_wav, &x, &n, &sr);
    if (err != ESP_OK) return err;
    if (sr != s_cfg.sample_rate)
        ESP_LOGW(TAG, "input sr=%d != model sr=%d (建议 48k 输入)", sr, s_cfg.sample_rate);

    float *y = (float *)psram_malloc((size_t)n * sizeof(float));
    if (!y) { free(x); return ESP_ERR_NO_MEM; }

    bnn_masknet_reset(s_net);
    bnn_masknet_perf_reset(s_net);              /* 清零诊断计时器 */
    bnn_ws_reset(NULL);                         /* 清空 workspace 峰值, 供基准计时 */

    int use_int8 = 0;
    if (opt && opt->use_int8) {
        if (s_int8_ready) {
            use_int8 = 1;
        } else {
            ESP_LOGW(TAG, "INT8 请求但不可用 (无 weights_i8 或未注入), 回退 F32");
        }
    }
    bnn_conv1d_set_int8_runtime(use_int8);

    int out_n = 0;
    int64_t t0 = esp_timer_get_time();
    int rc = bnn_masknet_process_audio(s_net, x, n, y, &out_n);
    bnn_conv1d_set_int8_runtime(0);
    int64_t dt_ms = (esp_timer_get_time() - t0) / 1000;
    size_t ws_peak_kb = bnn_ws_peak(NULL) / 1024;
    free(x);
    if (rc != 0) { ESP_LOGE(TAG, "process_audio rc=%d", rc); free(y); return ESP_FAIL; }

    /* ── 三段式诊断日志 (specfront / graph / synth) ─────────────────────── */
    bnn_masknet_perf_log(s_net);

    /* ── 更新统计累加器 ──────────────────────────────────────────────── */
    s_stats.n++;
    s_stats.sum_ms      += dt_ms;
    if (dt_ms < s_stats.min_ms)  s_stats.min_ms = dt_ms;
    if (dt_ms > s_stats.max_ms)  s_stats.max_ms = dt_ms;
    s_stats.sum_out_smp += out_n;
    if (ws_peak_kb > s_stats.ws_peak_kb) s_stats.ws_peak_kb = ws_peak_kb;

    /* ── 本次推理日志 ────────────────────────────────────────────────── */
    double rt_factor = (dt_ms > 0) ?
        (1000.0 * out_n / s_cfg.sample_rate) / (double)dt_ms : 0.0;
    ESP_LOGI(TAG, "[infer#%"PRIu32"] %s  %lldms  %.2fx RT  ws=%zuKB%s",
             s_stats.n, instrument, dt_ms, rt_factor, ws_peak_kb,
             use_int8 ? " INT8" : "");

    /* 后处理: 先变调, 再增益/削波 (顺序与 PC 端 render_instrument 一致) */
    if (opt) {
        if (opt->pitch_semitones != 0.0f)
            audio_post_pitch_shift(y, out_n, s_cfg.sample_rate, opt->pitch_semitones,
                                   s_cfg.n_fft, s_cfg.hop);
        if (opt->gain_db != 0.0f || opt->clip_mode != AUDIO_CLIP_LIMIT)
            audio_post_gain_clip(y, out_n, opt->gain_db, opt->clip_mode);
    }

    err = wav_write_mono_f32(out_wav, y, out_n, s_cfg.sample_rate);
    free(y);
    ESP_LOGI(TAG, "%s -> %s  %d->%d smp  %" PRId64 "ms",
             in_wav, out_wav, n, out_n, dt_ms);
    return err;
}

int audio_xform_loaded(void) { return s_loaded; }

int audio_xform_int8_ready(void) { return s_loaded && s_int8_ready; }

void audio_xform_print_model(void)
{
    if (!s_loaded) { ESP_LOGW(TAG, "模型未加载"); return; }
    int num_inst = s_pkg.num_inst > 0 ? s_pkg.num_inst : 0;
    ESP_LOGI(TAG, "模型: sr=%d  n_fft=%d  hop=%d  n_mels=%d  乐器数=%d  emb_dim=%d  params=%zu",
             s_cfg.sample_rate, s_cfg.n_fft, s_cfg.hop, s_cfg.n_mels,
             num_inst, s_pkg.emb_dim, bnn_masknet_num_params(s_net));
    /* names 段: \n 分隔的乐器名 */
    if (s_pkg.names && s_pkg.names_len > 0) {
        const char *p = s_pkg.names;
        int id = 0, start = 0;
        for (int i = 0; i <= s_pkg.names_len; ++i) {
            if (i == s_pkg.names_len || p[i] == '\n') {
                if (i > start) ESP_LOGI(TAG, "  [%d] %.*s", id, i - start, p + start);
                id++; start = i + 1;
            }
        }
    }
}

void audio_xform_print_stats(void)
{
    if (s_stats.n == 0) {
        ESP_LOGI(TAG, "[stats] 尚无推理记录");
        return;
    }
    int64_t avg_ms = s_stats.sum_ms / (int64_t)s_stats.n;
    double  avg_rt = (avg_ms > 0) ?
        (1000.0 * (double)s_stats.sum_out_smp /
         (double)s_cfg.sample_rate / (double)s_stats.n) / (double)avg_ms : 0.0;
    const char *backend = bnn_op_get_backend() ? bnn_op_get_backend()->name : "?";

    ESP_LOGI(TAG, "┌─── 推理统计 (backend=%s) ─────────────────┐", backend);
    ESP_LOGI(TAG, "│  调用次数 : %" PRIu32 "                              │", s_stats.n);
    ESP_LOGI(TAG, "│  耗时 avg : %" PRId64 " ms                           │", avg_ms);
    ESP_LOGI(TAG, "│  耗时 min : %" PRId64 " ms                           │", s_stats.min_ms);
    ESP_LOGI(TAG, "│  耗时 max : %" PRId64 " ms                           │", s_stats.max_ms);
    ESP_LOGI(TAG, "│  实时比   : %.2fx RT                          │", avg_rt);
    ESP_LOGI(TAG, "│  WS peak  : %zu KB                            │", s_stats.ws_peak_kb);
    ESP_LOGI(TAG, "└────────────────────────────────────────────┘");
}

void audio_xform_reset_stats(void)
{
    s_stats.n           = 0;
    s_stats.sum_ms      = 0;
    s_stats.min_ms      = INT64_MAX;
    s_stats.max_ms      = 0;
    s_stats.sum_out_smp = 0;
    s_stats.ws_peak_kb  = 0;
}

void audio_xform_infer_progress(int *pct, int *frame, int *total)
{
    int f = 0, t = 0;
    if (s_net) bnn_masknet_get_progress(s_net, &f, &t);
    if (frame) *frame = f;
    if (total) *total = t;
    if (pct)   *pct   = (t > 0) ? (f * 100 / t) : 0;
}

esp_err_t audio_xform_debug(const char *in_wav, const char *instrument)
{
    if (!s_loaded || !s_net) { ESP_LOGE(TAG, "not initialized"); return ESP_ERR_INVALID_STATE; }
    int id = model_store_instrument_id(&s_pkg, instrument);
    if (id < 0) {
        ESP_LOGW(TAG, "instrument '%s' 不在模型中", instrument);
        return ESP_ERR_NOT_FOUND;
    }
    bnn_masknet_set_instrument(s_net, id);
    bnn_masknet_set_add_noise(s_net, 0);
    bnn_masknet_set_debug(s_net, 1);

    float *x = NULL;
    int n = 0, sr = 0;
    esp_err_t err = wav_read_mono_f32(in_wav, &x, &n, &sr);
    if (err != ESP_OK) return err;

    float *y = (float *)psram_malloc((size_t)n * sizeof(float));
    if (!y) { free(x); return ESP_ERR_NO_MEM; }

    bnn_masknet_reset(s_net);
    int out_n = 0;
    int rc = bnn_masknet_process_audio(s_net, x, n, y, &out_n);
    free(x);
    free(y);
    if (rc != 0) { ESP_LOGE(TAG, "debug process_audio rc=%d", rc); return ESP_FAIL; }
    ESP_LOGI(TAG, "debug 完成: %s -> %s (%d smp)", in_wav, instrument, out_n);
    return ESP_OK;
}

void audio_xform_deinit(void)
{
    if (s_net) { bnn_masknet_destroy(s_net); s_net = NULL; }
    model_store_free(&s_pkg);
    if (s_ws_sram_buf) {
        heap_caps_free(s_ws_sram_buf);
        s_ws_sram_buf = NULL;
    }
    s_loaded = 0;
}
