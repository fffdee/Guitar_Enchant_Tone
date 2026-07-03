/* I2S 实时推理流水线实现。
 *
 * 数据流 (块级管线):
 *   I2S RX (16bit) -> int16→float -> 输入环形缓冲 -> 累积 T*hop 采样
 *   -> bnn_masknet_process_audio(整块) -> 输出环形缓冲 -> 交叉淡入淡出
 *   -> [后处理/效果器链] -> float→int16 -> I2S TX
 *
 * 块大小: T = block_frames (64帧), C = 4帧上下文重叠
 *       每块输入 = T*hop (16384), 新输入 = (T-C)*hop (15360)
 *       每块输出 ≈ T*hop, 丢弃前 C*hop (重叠) 后新输出 ≈ (T-C)*hop
 * 延迟: T*hop/sr ≈ 341ms (首块累积时间)
 *
 * 旁通模式: int16→float→直通→float→int16 (极简通道, ~20ms 延迟)
 */
#include "i2s_xform.h"
#include "i2s_driver.h"
#include "audio_xform.h"
#include "audio_post.h"
#include "fx_chain.h"
#include "model_store.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "bnn_model/bnn_masknet.h"
#include "bnn_layer/bnn_xform_layers.h"
#include <inttypes.h>

static const char *TAG = "i2s_xform";

#define I2S_TASK_STACK   8192
#define I2S_TASK_PRIO    7
#define I2S_TASK_CORE    1

/* CNN 非因果左侧感受野帧数 (bnn_masknet.c BNN_MASKNET_CTX_FRAMES) */
#define CTX_FRAMES       4

/* 环形缓冲区大小: 2× 完整块 = 32768 floats ≈ 128KB */
#define RING_MULT        2

/* 运行时状态 */
static struct {
    TaskHandle_t      task;
    SemaphoreHandle_t lock;
    i2s_xform_cfg_t  cfg;
    volatile int      running;
    volatile int      stop_req;
    volatile int      mode_switch;  /* 请求模式切换 (旁通↔推理) */
    /* 统计 */
    uint32_t          frames;
    uint32_t          blocks_infer; /* 推理块数 */
    int64_t           sum_us;
    int64_t           min_us;
    int64_t           max_us;
    int               underruns;
    int               overruns;
} s_rt;

static inline float i16_to_f(int16_t s) { return (float)s / 32768.0f; }
static inline int16_t f_to_i16(float f) {
    if (f >= 1.0f)  return 32767;
    if (f <= -1.0f) return -32768;
    return (int16_t)(int32_t)(f * 32768.0f);
}

/* ── 环形缓冲辅助 ───────────────────────────────────────────── */
static inline int ring_len(int wr, int rd, int mask) {
    return (wr - rd) & mask;
}
static inline void ring_write(float *r, int *wr, int mask, float v) {
    r[*wr] = v;
    *wr = (*wr + 1) & mask;
}
static inline float ring_read(const float *r, int *rd, int mask) {
    float v = r[*rd];
    *rd = (*rd + 1) & mask;
    return v;
}

/* ── 逐帧直通 (旁通模式) ──────────────────────────────────── */
static void bypass_loop(int hop, int sr, int16_t *i2s_buf)
{
    ESP_LOGI(TAG, "旁通直通模式, hop=%d sr=%d", hop, sr);
    while (!s_rt.stop_req) {
        int64_t t0 = esp_timer_get_time();

        /* 检查是否收到模式切换请求 */
        xSemaphoreTake(s_rt.lock, 0);
        int sw = s_rt.mode_switch;
        if (sw) s_rt.mode_switch = 0;
        xSemaphoreGive(s_rt.lock);
        if (sw) { ESP_LOGI(TAG, "旁通→推理 模式切换"); return; }

        /* 读 */
        int got = i2s_driver_read(i2s_buf, hop, 100);
        if (got < hop) { s_rt.underruns++; for (int i = got; i < hop; i++) i2s_buf[i] = 0; }

        /* 写 */
        int written = i2s_driver_write(i2s_buf, hop, 10);
        if (written < hop) s_rt.overruns++;

        s_rt.frames++;

        /* 速率对齐 */
        int64_t dur = (int64_t)hop * 1000000LL / sr;
        int64_t et = esp_timer_get_time() - t0;
        if (et < dur) esp_rom_delay_us((uint32_t)(dur - et));
    }
}

/* ── 块级推理管线 ─────────────────────────────────────────── */
static void inference_loop(int hop, int sr, int T, int16_t *i2s_buf)
{
    const int C    = CTX_FRAMES;
    const int BLK  = T * hop;             /* 完整块输入: 16384 采样 */
    const int CTX  = C * hop;             /* 上下文重叠: 1024 采样 */
    const int NEW  = BLK - CTX;           /* 每块新输入: 15360 采样 */

    /* 环形缓冲 (2倍大小以安全) */
    const int ring_sz  = BLK * RING_MULT;
    const int ring_msk = ring_sz - 1;

    float *in_ring  = (float *)heap_caps_malloc((size_t)ring_sz * sizeof(float),
                                                 MALLOC_CAP_SPIRAM);
    float *out_ring = (float *)heap_caps_malloc((size_t)ring_sz * sizeof(float),
                                                 MALLOC_CAP_SPIRAM);
    float *proc_in  = (float *)heap_caps_malloc((size_t)BLK * sizeof(float),
                                                 MALLOC_CAP_SPIRAM);
    float *proc_out = (float *)heap_caps_malloc((size_t)BLK * sizeof(float),
                                                 MALLOC_CAP_SPIRAM);
    /* 跨块上下文保存 */
    float *prev_in  = (float *)heap_caps_malloc((size_t)CTX * sizeof(float),
                                                 MALLOC_CAP_SPIRAM);
    float *prev_out = (float *)heap_caps_malloc((size_t)CTX * sizeof(float),
                                                 MALLOC_CAP_SPIRAM);

    if (!in_ring || !out_ring || !proc_in || !proc_out || !prev_in || !prev_out) {
        ESP_LOGE(TAG, "推理管线缓冲分配失败");
        goto infer_cleanup;
    }

    int in_wr = 0, in_rd = 0, in_smp = 0;
    int out_wr = 0, out_rd = 0, out_smp = 0;
    int block_idx = 0;
    int prev_out_valid = 0;  /* prev_out 缓冲区是否有有效数据 */

    enum { S_PRIME, S_INFER, S_PLAY, S_FINISH } state = S_PRIME;

    ESP_LOGI(TAG, "块级推理管线: T=%d C=%d hop=%d  block=%d ctx=%d new=%d  latency~%.0fms",
             T, C, hop, BLK, CTX, NEW, 1000.0f * (float)BLK / (float)sr);

    while (!s_rt.stop_req) {
        int64_t iter_start = esp_timer_get_time();

        /* ─ 检查模式切换 ─ */
        xSemaphoreTake(s_rt.lock, 0);
        int sw = s_rt.mode_switch;
        if (sw) s_rt.mode_switch = 0;
        xSemaphoreGive(s_rt.lock);
        if (sw) { ESP_LOGI(TAG, "推理→旁通 模式切换"); goto infer_cleanup; }

        /* ─ 1. 读 I2S RX → 入环 ─ */
        int got = i2s_driver_read(i2s_buf, hop, 100);
        if (got < hop) {
            s_rt.underruns++;
            for (int i = (got > 0 ? got : 0); i < hop; i++) i2s_buf[i] = 0;
        }
        for (int i = 0; i < hop; i++)
            ring_write(in_ring, &in_wr, ring_msk, i16_to_f(i2s_buf[i]));
        in_smp += hop;

        /* ─ 2. 状态机 ─ */
        if (state == S_PRIME && in_smp >= BLK) {
            state = S_INFER;
        }
        if (state == S_PLAY && in_smp >= NEW) {
            state = S_INFER;
        }

        if (state == S_INFER) {
            /* ▸ 准备推理输入 */
            if (block_idx == 0) {
                /* 首块: 直接用 in_ring 中 BLK 个采样 */
                for (int i = 0; i < BLK; i++)
                    proc_in[i] = ring_read(in_ring, &in_rd, ring_msk);
                in_smp -= BLK;
            } else {
                /* 后续块: [prev_in (CTX)] + [new from ring (NEW)] */
                memcpy(proc_in, prev_in, (size_t)CTX * sizeof(float));
                for (int i = 0; i < NEW; i++)
                    proc_in[CTX + i] = ring_read(in_ring, &in_rd, ring_msk);
                in_smp -= NEW;
            }

            /* 保存本块尾部 CTX 采样给下一块 */
            {
                int base = BLK - CTX;
                for (int i = 0; i < CTX; i++)
                    prev_in[i] = proc_in[base + i];
            }

            /* ▸ 运行推理 */
            int out_n = 0;
            int64_t t_infer = esp_timer_get_time();
            int rc = bnn_masknet_process_audio(s_net, proc_in, BLK, proc_out, &out_n);
            int64_t infer_us = esp_timer_get_time() - t_infer;

            if (rc != 0 || out_n <= 0) {
                ESP_LOGW(TAG, "推理失败 rc=%d out_n=%d (block %d)", rc, out_n, block_idx);
                /* 失败时不做输出, 但不清空状态 (后续继续尝试) */
                state = S_PLAY;
            } else {

            /* ▸ 推入输出环 */
            if (block_idx == 0) {
                /* 首块: 全部推入 */
                for (int i = 0; i < out_n; i++)
                    ring_write(out_ring, &out_wr, ring_msk, proc_out[i]);
                out_smp += out_n;
            } else {
                /* 后续块: 前 CTX 采样与 prev_out 交叉淡入淡出, 其余直接推入 */
                int xfade_len = (CTX < out_n) ? CTX : out_n;
                if (prev_out_valid) {
                    float inv = 1.0f / (float)xfade_len;
                    for (int i = 0; i < xfade_len; i++) {
                        float a = (float)i * inv;           /* 0→1 :  新块权重 */
                        float b = 1.0f - a;                 /* 1→0 : 旧块权重 */
                        float v = b * prev_out[i] + a * proc_out[i];
                        ring_write(out_ring, &out_wr, ring_msk, v);
                    }
                } else {
                    for (int i = 0; i < xfade_len; i++)
                        ring_write(out_ring, &out_wr, ring_msk, proc_out[i]);
                }
                out_smp += xfade_len;

                int tail_len = out_n - CTX;
                if (tail_len > 0) {
                    for (int i = 0; i < tail_len; i++)
                        ring_write(out_ring, &out_wr, ring_msk, proc_out[CTX + i]);
                    out_smp += tail_len;
                }
            }

            /* 保存本块输出尾部 CTX 采样 (下一块交叉淡入淡出用) */
            {
                int out_tail_base = (out_n > CTX) ? (out_n - CTX) : 0;
                int nsave = (out_n > CTX) ? CTX : out_n;
                for (int i = 0; i < nsave; i++)
                    prev_out[i] = proc_out[out_tail_base + i];
                prev_out_valid = nsave;
            }

            block_idx++;
            s_rt.blocks_infer = (uint32_t)block_idx;
            s_rt.sum_us += infer_us;
            if (infer_us < s_rt.min_us) s_rt.min_us = infer_us;
            if (infer_us > s_rt.max_us) s_rt.max_us = infer_us;

            state = S_PLAY;
            }  /* end else (推理成功) */

        }  /* end S_INFER */

        /* ─ 3. 从输出环取 hop 采样 → I2S TX ─ */
        float tx_hop[256];  /* 最大 hop */
        if (out_smp >= hop) {
            for (int i = 0; i < hop; i++)
                tx_hop[i] = ring_read(out_ring, &out_rd, ring_msk);
            out_smp -= hop;
        } else {
            memset(tx_hop, 0, (size_t)hop * sizeof(float));
        }

        /* ─ 4. 后处理 + 效果器 ─ */
        {
            xSemaphoreTake(s_rt.lock, portMAX_DELAY);
            i2s_xform_cfg_t cfg = s_rt.cfg;
            xSemaphoreGive(s_rt.lock);
            if (!cfg.bypass) {
                if (cfg.pitch_semitones != 0.0f)
                    audio_post_pitch_shift(tx_hop, hop, sr, cfg.pitch_semitones,
                                           s_cfg.n_fft, s_cfg.hop);
                if (cfg.gain_db != 0.0f || cfg.clip_mode != AUDIO_CLIP_LIMIT)
                    audio_post_gain_clip(tx_hop, hop, cfg.gain_db, cfg.clip_mode);
            }
            if (cfg.post_fx)
                fx_chain_process(tx_hop, hop);
        }

        /* ─ 5. float → int16 → I2S TX ─ */
        for (int i = 0; i < hop; i++)
            i2s_buf[i] = f_to_i16(tx_hop[i]);
        int written = i2s_driver_write(i2s_buf, hop, 10);
        if (written < hop) s_rt.overruns++;

        s_rt.frames++;

        /* ─ 6. 速率对齐 ─ */
        int64_t frame_dur_us = (int64_t)hop * 1000000LL / sr;
        int64_t cycle_us = esp_timer_get_time() - iter_start;
        if (cycle_us < frame_dur_us)
            esp_rom_delay_us((uint32_t)(frame_dur_us - cycle_us));
    }

infer_cleanup:
    if (in_ring)  heap_caps_free(in_ring);
    if (out_ring) heap_caps_free(out_ring);
    if (proc_in)  heap_caps_free(proc_in);
    if (proc_out) heap_caps_free(proc_out);
    if (prev_in)  heap_caps_free(prev_in);
    if (prev_out) heap_caps_free(prev_out);
}

/* ── 主任务入口 ────────────────────────────────────────────── */
static void rt_task(void *arg)
{
    (void)arg;
    int hop = s_cfg.hop;
    int sr  = s_cfg.sample_rate;

    int16_t *i2s_buf = (int16_t *)heap_caps_malloc((size_t)hop * sizeof(int16_t),
                                                    MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    if (!i2s_buf) {
        ESP_LOGE(TAG, "I2S 缓冲分配失败");
        goto cleanup;
    }

    /* 设置乐器 (仅推理模式) */
    int bypass = s_rt.cfg.bypass;
    if (!bypass) {
        int inst_id = model_store_instrument_id(&s_pkg, s_rt.cfg.instrument);
        if (inst_id < 0) {
            ESP_LOGE(TAG, "乐器 '%s' 不在模型中", s_rt.cfg.instrument);
            goto cleanup;
        }
        bnn_masknet_set_instrument(s_net, inst_id);
        bnn_masknet_set_add_noise(s_net, s_rt.cfg.add_noise);
        bnn_masknet_reset(s_net);
    }

    int use_int8 = 0;
    if (!bypass && s_rt.cfg.use_int8) {
        if (audio_xform_int8_ready()) use_int8 = 1;
        else ESP_LOGW(TAG, "INT8 不可用, F32");
    }
    bnn_conv1d_set_int8_runtime(use_int8);

    esp_err_t err = i2s_driver_start();
    if (err != ESP_OK) { ESP_LOGE(TAG, "I2S 启动失败"); goto cleanup; }

    ESP_LOGI(TAG, "实时管线启动: %s  hop=%d sr=%d%s%s",
             bypass ? "(旁通)" : s_rt.cfg.instrument,
             hop, sr, use_int8 ? " INT8" : "",
             bypass ? "" : (s_rt.cfg.post_fx ? " +FX" : ""));
    s_rt.running = 1;

    /* 主循环: 旁通 / 推理 交替 */
    while (!s_rt.stop_req) {
        if (s_rt.cfg.bypass) {
            bypass_loop(hop, sr, i2s_buf);
        } else {
            /* 每次进入推理模式前重置模型流式状态 */
            if (s_net) {
                bnn_masknet_reset(s_net);
                bnn_masknet_set_instrument(s_net,
                    model_store_instrument_id(&s_pkg, s_rt.cfg.instrument));
            }
            int T = bnn_masknet_block_frames(s_net);
            if (T <= 0) { ESP_LOGE(TAG, "block_frames=0"); break; }
            inference_loop(hop, sr, T, i2s_buf);
        }
    }

    i2s_driver_stop();
    s_rt.running = 0;
    ESP_LOGI(TAG, "实时推理已停止: %" PRIu32 "帧 块=%" PRIu32 " avg=%" PRId64 "us  under=%d over=%d",
             s_rt.frames, s_rt.blocks_infer,
             s_rt.blocks_infer > 0 ? s_rt.sum_us / (int64_t)s_rt.blocks_infer : 0,
             s_rt.underruns, s_rt.overruns);

cleanup:
    bnn_conv1d_set_int8_runtime(0);
    if (i2s_buf) heap_caps_free(i2s_buf);
    s_rt.running = 0;
    s_rt.task = NULL;
    vTaskDelete(NULL);
}

esp_err_t i2s_xform_start(const i2s_xform_cfg_t *cfg)
{
    if (!cfg) return ESP_ERR_INVALID_ARG;
    if (s_rt.running) {
        ESP_LOGW(TAG, "已在运行, 请先停止");
        return ESP_ERR_INVALID_STATE;
    }
    /* 旁通模式不依赖模型; 非旁通模式需要模型已加载 */
    if (!cfg->bypass && !audio_xform_loaded()) {
        ESP_LOGE(TAG, "模型未加载 (非旁通模式需要模型)");
        return ESP_ERR_INVALID_STATE;
    }
    if (!i2s_driver_is_ready()) {
        ESP_LOGE(TAG, "I2S 未初始化");
        return ESP_ERR_INVALID_STATE;
    }

    /* 重置状态 */
    memset(&s_rt, 0, sizeof(s_rt));
    s_rt.min_us = INT64_MAX;
    s_rt.mode_switch = 0;
    s_rt.lock = xSemaphoreCreateMutex();
    if (!s_rt.lock) return ESP_ERR_NO_MEM;
    memcpy(&s_rt.cfg, cfg, sizeof(s_rt.cfg));

    BaseType_t ok = xTaskCreatePinnedToCore(rt_task, "i2s_xform", I2S_TASK_STACK,
                                            NULL, I2S_TASK_PRIO, &s_rt.task, I2S_TASK_CORE);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "任务创建失败");
        vSemaphoreDelete(s_rt.lock);
        s_rt.lock = NULL;
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t i2s_xform_stop(void)
{
    if (!s_rt.running && !s_rt.task) return ESP_OK;
    s_rt.stop_req = 1;
    /* 等待任务退出 (最多 2 秒) */
    for (int i = 0; i < 200 && s_rt.running; i++)
        vTaskDelay(pdMS_TO_TICKS(10));
    if (s_rt.lock) { vSemaphoreDelete(s_rt.lock); s_rt.lock = NULL; }
    return ESP_OK;
}

int i2s_xform_running(void)
{
    return s_rt.running;
}

esp_err_t i2s_xform_set_instrument(const char *instrument)
{
    if (!s_rt.running || !s_net) return ESP_ERR_INVALID_STATE;
    int id = model_store_instrument_id(&s_pkg, instrument);
    if (id < 0) return ESP_ERR_NOT_FOUND;
    bnn_masknet_set_instrument(s_net, id);
    bnn_masknet_reset(s_net);   /* 切换乐器 = 全新管线, 重置流式状态 */
    xSemaphoreTake(s_rt.lock, portMAX_DELAY);
    strncpy(s_rt.cfg.instrument, instrument, sizeof(s_rt.cfg.instrument) - 1);
    s_rt.cfg.bypass = 0;        /* 切乐器自动退出旁通 */
    s_rt.mode_switch = 1;       /* 重建推理管线 */
    xSemaphoreGive(s_rt.lock);
    ESP_LOGI(TAG, "乐器切换: %s (管线重启)", instrument);
    return ESP_OK;
}

esp_err_t i2s_xform_set_post(float pitch_semitones, float gain_db,
                              audio_clip_mode_t clip_mode, int add_noise)
{
    if (!s_rt.running) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(s_rt.lock, portMAX_DELAY);
    s_rt.cfg.pitch_semitones = pitch_semitones;
    s_rt.cfg.gain_db = gain_db;
    s_rt.cfg.clip_mode = clip_mode;
    s_rt.cfg.add_noise = add_noise;
    xSemaphoreGive(s_rt.lock);
    if (!s_rt.cfg.bypass && s_net)
        bnn_masknet_set_add_noise(s_net, add_noise);
    return ESP_OK;
}

esp_err_t i2s_xform_set_bypass(int bypass)
{
    if (!s_rt.running) return ESP_ERR_INVALID_STATE;
    int target = bypass ? 1 : 0;
    if (s_rt.cfg.bypass == target) return ESP_OK;  /* 未变化 */
    xSemaphoreTake(s_rt.lock, portMAX_DELAY);
    s_rt.cfg.bypass = target;
    s_rt.mode_switch = 1;   /* 通知 rt_task 切换模式 */
    xSemaphoreGive(s_rt.lock);
    ESP_LOGI(TAG, "音色转换 %s (管线将重启)", bypass ? "已旁通" : "已启用");
    return ESP_OK;
}

esp_err_t i2s_xform_set_post_fx(int enabled)
{
    if (!s_rt.running) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(s_rt.lock, portMAX_DELAY);
    s_rt.cfg.post_fx = enabled ? 1 : 0;
    xSemaphoreGive(s_rt.lock);
    ESP_LOGI(TAG, "效果器链后处理 %s", enabled ? "已启用" : "已关闭");
    return ESP_OK;
}

void i2s_xform_status(char *buf, size_t n)
{
    if (!buf || n == 0) return;
    if (!s_rt.running) {
        snprintf(buf, n, "I2S 实时推理: 停止");
        return;
    }
    int64_t avg_us = s_rt.blocks_infer > 0 ? s_rt.sum_us / (int64_t)s_rt.blocks_infer : 0;
    double rt_factor = (avg_us > 0 && s_rt.blocks_infer > 0) ?
        (1000.0 * (double)s_cfg.hop * bnn_masknet_block_frames(s_net) / s_cfg.sample_rate)
         / (avg_us / 1000.0) : 0.0;
    snprintf(buf, n,
             "I2S 实时: %s%s  %" PRIu32 "帧 块=%" PRIu32 " avg=%" PRId64 "us/块(%.1fxRT)"
             "  min=%" PRId64 "us  max=%" PRId64 "us  under=%d  over=%d",
             s_rt.cfg.bypass ? "(旁通)" : s_rt.cfg.instrument,
             s_rt.cfg.post_fx ? " +FX" : "",
             s_rt.frames, s_rt.blocks_infer, avg_us, rt_factor,
             s_rt.min_us < INT64_MAX ? s_rt.min_us : 0,
             s_rt.max_us, s_rt.underruns, s_rt.overruns);
}
