#include "audio_post.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "bnn_op/bnn_dsp.h"

static const char *TAG = "audio_post";

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void *ps_malloc(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    return p ? p : malloc(sz);
}

void audio_post_gain_clip(float *y, int n, float gain_db, audio_clip_mode_t mode)
{
    if (!y || n <= 0) return;
    float g = powf(10.0f, gain_db / 20.0f);
    if (mode == AUDIO_CLIP_SOFT) {
        for (int i = 0; i < n; ++i) y[i] = tanhf(y[i] * g);
    } else if (mode == AUDIO_CLIP_HARD) {
        for (int i = 0; i < n; ++i) {
            float v = y[i] * g;
            y[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
        }
    } else { /* limit: 先增益, 峰值超 0.999 整体回缩 */
        float peak = 0.0f;
        for (int i = 0; i < n; ++i) { float v = fabsf(y[i] * g); if (v > peak) peak = v; }
        float s = (peak > 0.999f) ? (g * 0.999f / peak) : g;
        for (int i = 0; i < n; ++i) y[i] *= s;
    }
}

/* 相位声码器: 流式 STFT -> 相位累加(合成 hop=round(hop*ratio)) -> OLA -> 线性重采样回原长.
 * 复现 mask/audio.py:pitch_shift_semitones 的核心 (重采样用线性插值近似 FFT 重采样). */
int audio_post_pitch_shift(float *y, int n, int sr, float semitones, int n_fft, int hop)
{
    (void)sr;
    if (!y || n < n_fft || fabsf(semitones) < 1e-6f) return 0;
    const bnn_dsp_backend_t *dsp = bnn_dsp_get_backend();
    if (!dsp || !dsp->rfft || !dsp->irfft) { ESP_LOGE(TAG, "no FFT backend"); return -1; }

    const float ratio = powf(2.0f, semitones / 12.0f);
    const int   Hs    = (int)lroundf((float)hop * ratio); /* 合成 hop */
    const int   nb    = n_fft / 2 + 1;
    const int   pad   = n_fft / 2;                        /* center 填充 */
    const int   n_frames = 1 + n / hop;                   /* 分析帧数 (padded) */
    const int   Ls_full  = (n_frames - 1) * (Hs < 1 ? 1 : Hs) + n_fft;

    /* 缓冲: 窗/帧/谱/相位放内部 RAM; OLA 大缓冲放 PSRAM */
    float *win   = (float *)malloc(sizeof(float) * n_fft);
    float *frame = (float *)malloc(sizeof(float) * n_fft);
    float *spec  = (float *)malloc(sizeof(float) * 2 * nb);
    float *mag   = (float *)malloc(sizeof(float) * nb);
    float *acc   = (float *)malloc(sizeof(float) * nb);
    float *prevp = (float *)malloc(sizeof(float) * nb);
    float *omega = (float *)malloc(sizeof(float) * nb);
    float *xpad  = (float *)ps_malloc(sizeof(float) * (size_t)(n + 2 * pad));
    float *str   = (float *)ps_malloc(sizeof(float) * (size_t)Ls_full);
    float *wsum  = (float *)ps_malloc(sizeof(float) * (size_t)Ls_full);
    if (!win || !frame || !spec || !mag || !acc || !prevp || !omega || !xpad || !str || !wsum) {
        ESP_LOGE(TAG, "pitch_shift OOM");
        free(win); free(frame); free(spec); free(mag); free(acc); free(prevp); free(omega);
        free(xpad); free(str); free(wsum);
        return -1;
    }

    for (int k = 0; k < n_fft; ++k) win[k] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * k / n_fft);
    for (int b = 0; b < nb; ++b) omega[b] = 2.0f * (float)M_PI * hop * b / n_fft;

    /* center 反射填充 */
    for (int i = 0; i < n; ++i) xpad[pad + i] = y[i];
    for (int i = 0; i < pad; ++i) {
        xpad[pad - 1 - i] = y[(i + 1 < n) ? i + 1 : n - 1];
        xpad[pad + n + i] = y[(n - 2 - i >= 0) ? n - 2 - i : 0];
    }
    memset(str, 0, sizeof(float) * (size_t)Ls_full);
    memset(wsum, 0, sizeof(float) * (size_t)Ls_full);

    const float scale = (float)(Hs < 1 ? 1 : Hs) / (float)hop;
    for (int t = 0; t < n_frames; ++t) {
        const float *seg = xpad + t * hop;
        for (int k = 0; k < n_fft; ++k) frame[k] = seg[k] * win[k];
        dsp->rfft(frame, spec, n_fft);
        for (int b = 0; b < nb; ++b) {
            float re = spec[2 * b], im = spec[2 * b + 1];
            float m  = sqrtf(re * re + im * im);
            float ph = atan2f(im, re);
            mag[b] = m;
            if (t == 0) { acc[b] = ph; }
            else {
                float dphi = ph - prevp[b] - omega[b];
                dphi = dphi - 2.0f * (float)M_PI * roundf(dphi / (2.0f * (float)M_PI));
                acc[b] += scale * (omega[b] + dphi);
            }
            prevp[b] = ph;
            spec[2 * b]     = m * cosf(acc[b]);
            spec[2 * b + 1] = m * sinf(acc[b]);
        }
        dsp->irfft(spec, frame, n_fft);   /* 已 /n_fft 归一 */
        int s0 = t * (Hs < 1 ? 1 : Hs);
        for (int k = 0; k < n_fft; ++k) {
            str[s0 + k]  += frame[k] * win[k];
            wsum[s0 + k] += win[k] * win[k];
        }
    }
    for (int i = 0; i < Ls_full; ++i) if (wsum[i] > 1e-8f) str[i] /= wsum[i];

    /* 去 center 填充 -> 拉伸段长度 Ls, 再线性重采样回 n */
    int Ls = Ls_full - 2 * pad;
    if (Ls < 1) Ls = 1;
    const float *src = str + pad;
    double rms_in = 0.0, rms_out = 0.0;
    for (int i = 0; i < n; ++i) rms_in += (double)y[i] * y[i];
    for (int i = 0; i < n; ++i) {
        float pos = (float)i * (float)(Ls - 1) / (float)(n - 1 > 0 ? n - 1 : 1);
        int   i0  = (int)pos;
        float fr  = pos - i0;
        float a   = src[i0 < Ls ? i0 : Ls - 1];
        float b   = src[(i0 + 1) < Ls ? i0 + 1 : Ls - 1];
        float v   = a + (b - a) * fr;
        y[i] = v;
        rms_out += (double)v * v;
    }
    /* RMS 保持 (变调不应改变响度), 限幅保护 */
    if (rms_in > 1e-12 && rms_out > 1e-12) {
        float gain = (float)sqrt(rms_in / rms_out);
        if (gain > 8.0f) gain = 8.0f; else if (gain < 0.125f) gain = 0.125f;
        for (int i = 0; i < n; ++i) y[i] *= gain;
    }
    float peak = 0.0f;
    for (int i = 0; i < n; ++i) { float a = fabsf(y[i]); if (a > peak) peak = a; }
    if (peak > 0.999f) { float s = 0.999f / peak; for (int i = 0; i < n; ++i) y[i] *= s; }

    free(win); free(frame); free(spec); free(mag); free(acc); free(prevp); free(omega);
    free(xpad); free(str); free(wsum);
    ESP_LOGI(TAG, "pitch %+.1f semitones (ratio %.3f, Hs=%d) done", semitones, ratio, Hs);
    return 0;
}
