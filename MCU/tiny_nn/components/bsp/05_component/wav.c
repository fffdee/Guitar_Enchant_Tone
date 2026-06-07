#include "wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "wav";

static void *psram_malloc(size_t sz)
{
    void *p = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    return p ? p : malloc(sz);   /* 无 PSRAM 时回退内部 RAM */
}

#pragma pack(push, 1)
typedef struct {
    char riff[4]; uint32_t size; char wave[4];
} riff_hdr_t;
typedef struct {
    char id[4]; uint32_t size;
} chunk_hdr_t;
typedef struct {
    uint16_t fmt; uint16_t channels; uint32_t sample_rate;
    uint32_t byte_rate; uint16_t block_align; uint16_t bits;
} fmt_chunk_t;
#pragma pack(pop)

esp_err_t wav_read_mono_f32(const char *path, float **out, int *n_samples, int *sample_rate)
{
    if (!path || !out || !n_samples) return ESP_ERR_INVALID_ARG;
    *out = NULL; *n_samples = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { ESP_LOGE(TAG, "open %s failed", path); return ESP_FAIL; }

    riff_hdr_t rh;
    if (fread(&rh, sizeof(rh), 1, f) != 1 || memcmp(rh.riff, "RIFF", 4) || memcmp(rh.wave, "WAVE", 4)) {
        ESP_LOGE(TAG, "not RIFF/WAVE"); fclose(f); return ESP_FAIL;
    }

    fmt_chunk_t fmt = {0};
    int have_fmt = 0;
    long data_pos = -1; uint32_t data_size = 0;
    chunk_hdr_t ch;
    while (fread(&ch, sizeof(ch), 1, f) == 1) {
        if (!memcmp(ch.id, "fmt ", 4)) {
            uint32_t rd = ch.size < sizeof(fmt) ? ch.size : sizeof(fmt);
            if (fread(&fmt, rd, 1, f) != 1) break;
            have_fmt = 1;
            if (ch.size > rd) fseek(f, ch.size - rd, SEEK_CUR);
        } else if (!memcmp(ch.id, "data", 4)) {
            data_pos = ftell(f); data_size = ch.size;
            fseek(f, ch.size, SEEK_CUR);
        } else {
            fseek(f, ch.size, SEEK_CUR);
        }
        if (ch.size & 1) fseek(f, 1, SEEK_CUR);   /* 字块按偶数对齐 */
    }
    if (!have_fmt || data_pos < 0) { ESP_LOGE(TAG, "no fmt/data"); fclose(f); return ESP_FAIL; }

    int bps = fmt.bits / 8;
    int ch_n = fmt.channels ? fmt.channels : 1;
    if (bps < 1 || bps > 4) { ESP_LOGE(TAG, "unsupported bits=%u", (unsigned)fmt.bits); fclose(f); return ESP_FAIL; }
    int frames = (int)(data_size / (uint32_t)(bps * ch_n));
    if (frames <= 0) { fclose(f); return ESP_FAIL; }

    float *buf = (float *)psram_malloc((size_t)frames * sizeof(float));
    if (!buf) { ESP_LOGE(TAG, "OOM %d frames", frames); fclose(f); return ESP_ERR_NO_MEM; }

    fseek(f, data_pos, SEEK_SET);
    const float invs = 1.0f / 2147483648.0f;   /* 归一到 [-1,1] (按 32-bit 满幅) */
    uint8_t raw[4 * 8];
    for (int i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int c = 0; c < ch_n; ++c) {
            if (fread(raw, bps, 1, f) != 1) { acc = 0; break; }
            int32_t v = 0;
            if (bps == 2) { v = (int32_t)(int16_t)(raw[0] | (raw[1] << 8)); v <<= 16; }
            else if (bps == 3) { v = (raw[0] << 8) | (raw[1] << 16) | (raw[2] << 24); }
            else if (bps == 4) { v = (int32_t)(raw[0] | (raw[1] << 8) | (raw[2] << 16) | (raw[3] << 24)); }
            else { v = ((int32_t)raw[0] - 128) << 24; }   /* 8-bit 无符号 */
            acc += (double)v * invs;
        }
        buf[i] = (float)(acc / ch_n);
    }
    fclose(f);
    *out = buf; *n_samples = frames;
    if (sample_rate) *sample_rate = (int)fmt.sample_rate;
    ESP_LOGI(TAG, "read %s: %d frames %" PRIu32 "Hz %dch %ubit", path, frames,
             fmt.sample_rate, ch_n, (unsigned)fmt.bits);
    return ESP_OK;
}

esp_err_t wav_write_mono_f32(const char *path, const float *data, int n_samples, int sample_rate)
{
    if (!path || !data || n_samples <= 0) return ESP_ERR_INVALID_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) { ESP_LOGE(TAG, "create %s failed", path); return ESP_FAIL; }

    uint32_t data_bytes = (uint32_t)n_samples * 2;
    riff_hdr_t rh = {{'R','I','F','F'}, 36 + data_bytes, {'W','A','V','E'}};
    chunk_hdr_t fch = {{'f','m','t',' '}, 16};
    fmt_chunk_t fmt = {1, 1, (uint32_t)sample_rate, (uint32_t)sample_rate * 2, 2, 16};
    chunk_hdr_t dch = {{'d','a','t','a'}, data_bytes};
    fwrite(&rh, sizeof(rh), 1, f);
    fwrite(&fch, sizeof(fch), 1, f);
    fwrite(&fmt, sizeof(fmt), 1, f);
    fwrite(&dch, sizeof(dch), 1, f);

    for (int i = 0; i < n_samples; ++i) {
        float x = data[i];
        if (x > 1.0f) x = 1.0f; else if (x < -1.0f) x = -1.0f;
        int16_t s = (int16_t)(x * 32767.0f);
        fwrite(&s, sizeof(s), 1, f);
    }
    fclose(f);
    ESP_LOGI(TAG, "wrote %s: %d frames %dHz 16bit mono", path, n_samples, sample_rate);
    return ESP_OK;
}
