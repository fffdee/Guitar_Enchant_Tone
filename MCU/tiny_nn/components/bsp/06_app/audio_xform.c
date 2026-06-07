#include "audio_xform.h"
#include "model_store.h"
#include "wav.h"
#include <stdlib.h>
#include <inttypes.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "bnn_model/bnn_masknet.h"
#include "bnn_frontend/bnn_mask_config.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include "bnn_op/bnn_dsp.h"
#include "dsp_accel.h"

static const char *TAG = "xform";
#define BLOCK_FRAMES 64

/* 把 tinynn 内部日志转到 ESP 串口 (MCU 默认 sink 为 no-op, 否则看不到失败原因) */
static void bnn_log_to_esp(int level, const char *msg, void *user)
{
    (void)user;
    if (level <= 1) ESP_LOGE("bnn", "%s", msg);
    else            ESP_LOGI("bnn", "%s", msg);
}

static model_pkg_t    s_pkg;
static bnn_masknet_t *s_net = NULL;
static bnn_mask_cfg_t s_cfg;
static int            s_loaded = 0;

static void *psram_malloc(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    return p ? p : malloc(sz);
}

/* tinynn 内存分配器: 大块(卷积权重~596KB / 展开矩阵)走 PSRAM, 小块走内部 RAM.
 * 阈值取 16KB: 让 FFT 临时缓冲(frame_tmp/spec/time 等 ~n_fft 大小)留在内部 RAM,
 * 避免 cfft/irfft 频繁随机访问 PSRAM 拖慢; 否则 596KB 权重在内部 RAM 必然 OOM. */
#define BNN_PSRAM_THRESHOLD (16 * 1024)
static void *bnn_psram_alloc(size_t sz)
{
    uint32_t caps = (sz >= BNN_PSRAM_THRESHOLD) ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
    void *p = heap_caps_malloc(sz, caps);
    if (!p) p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);  /* 回退 PSRAM */
    if (!p) p = malloc(sz);
    return p;
}
static void bnn_psram_free(void *p) { free(p); }

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

    esp_err_t err = model_store_load(model_path, &s_pkg);
    if (err != ESP_OK) return err;

    cfg_from_pkg(&s_cfg, &s_pkg);
    int num_inst = s_pkg.num_inst > 0 ? s_pkg.num_inst : 1;

    /* 接入 ESP32-P4 esp-dsp 单精度 FFT 后端 (替换默认 double-twiddle 软件实现) */
    if (dsp_accel_init(s_cfg.n_fft) == ESP_OK) {
        bnn_dsp_set_backend(dsp_accel_backend());
        ESP_LOGI(TAG, "FFT 后端: esp-dsp P4 (n_fft=%d)", s_cfg.n_fft);
    } else {
        ESP_LOGW(TAG, "esp-dsp 初始化失败, 回退默认 FFT (较慢)");
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

    float *x = NULL; int n = 0, sr = 0;
    esp_err_t err = wav_read_mono_f32(in_wav, &x, &n, &sr);
    if (err != ESP_OK) return err;
    if (sr != s_cfg.sample_rate)
        ESP_LOGW(TAG, "input sr=%d != model sr=%d (建议 48k 输入)", sr, s_cfg.sample_rate);

    float *y = (float *)psram_malloc((size_t)n * sizeof(float));
    if (!y) { free(x); return ESP_ERR_NO_MEM; }

    bnn_masknet_reset(s_net);
    int out_n = 0;
    int64_t t0 = esp_timer_get_time();
    int rc = bnn_masknet_process_audio(s_net, x, n, y, &out_n);
    int64_t dt_ms = (esp_timer_get_time() - t0) / 1000;
    free(x);
    if (rc != 0) { ESP_LOGE(TAG, "process_audio rc=%d", rc); free(y); return ESP_FAIL; }

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
    ESP_LOGI(TAG, "%s -> %s [%s] %d->%d smp, %" PRId64 "ms (%.1fx RT)",
             in_wav, out_wav, instrument, n, out_n, dt_ms,
             dt_ms > 0 ? (1000.0 * out_n / s_cfg.sample_rate) / dt_ms : 0.0);
    return err;
}

int audio_xform_loaded(void) { return s_loaded; }

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

void audio_xform_deinit(void)
{
    if (s_net) { bnn_masknet_destroy(s_net); s_net = NULL; }
    model_store_free(&s_pkg);
    s_loaded = 0;
}
