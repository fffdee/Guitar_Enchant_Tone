#include "model_store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "model_store";
#define TOC_ENTRY 48

/* ── INT8 迭代器 ──────────────────────────────────────────────────────────── */

void model_i8_iter_init(model_i8_iter_t *it, const model_pkg_t *pkg)
{
    memset(it, 0, sizeof(*it));
    if (!pkg || !pkg->weights_i8 || pkg->weights_i8_size < 16) return;

    const uint8_t *p = (const uint8_t *)pkg->weights_i8;
    /* 验证内部魔数 */
    uint32_t magic = (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
    if (magic != BNNW_I8_MAGIC) {
        ESP_LOGW("model_store", "weights_i8 magic mismatch (got %08" PRIx32 ")", magic);
        return;
    }
    uint32_t num_conv = (uint32_t)p[8] | ((uint32_t)p[9]<<8) | ((uint32_t)p[10]<<16) | ((uint32_t)p[11]<<24);
    it->p    = p + 16;   /* 跳过 16B 头 */
    it->end  = p + pkg->weights_i8_size;
    it->left = num_conv;
}

int model_i8_iter_next(model_i8_iter_t *it)
{
    if (!it || it->left == 0 || it->p + 16 > it->end) return 0;

    const uint8_t *p = it->p;
    /* 每层块头: Cout(u32) + Cin_K(u32) + nbytes(u32) + _rsv(u32) */
    uint32_t Cout   = (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
    uint32_t Cin_K  = (uint32_t)p[4]|((uint32_t)p[5]<<8)|((uint32_t)p[6]<<16)|((uint32_t)p[7]<<24);
    uint32_t nbytes = (uint32_t)p[8]|((uint32_t)p[9]<<8)|((uint32_t)p[10]<<16)|((uint32_t)p[11]<<24);
    p += 16;

    /* INT8 权重 */
    if (p + nbytes > it->end) return 0;
    it->W_i8  = (const int8_t *)p;
    it->Cout  = Cout;
    it->Cin_K = Cin_K;
    p += nbytes;

    /* 4 字节对齐填充 */
    uint32_t pad = (4 - (nbytes % 4)) % 4;
    p += pad;

    /* float32 scale [Cout] */
    size_t scale_sz = sizeof(float) * Cout;
    if (p + scale_sz > it->end) return 0;
    it->scale = (const float *)p;
    p += scale_sz;

    it->p = p;
    it->left--;
    return 1;
}

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

esp_err_t model_store_load(const char *path, model_pkg_t *pkg)
{
    if (!path || !pkg) return ESP_ERR_INVALID_ARG;
    memset(pkg, 0, sizeof(*pkg));

    FILE *f = fopen(path, "rb");
    if (!f) { ESP_LOGE(TAG, "open %s failed", path); return ESP_FAIL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 16) { fclose(f); return ESP_FAIL; }

    uint8_t *buf = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    if (!buf) buf = malloc(sz);
    if (!buf) { fclose(f); ESP_LOGE(TAG, "OOM %ld", (long)sz); return ESP_ERR_NO_MEM; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return ESP_FAIL; }
    fclose(f);

    if (rd_u32(buf) != MODEL_PKG_MAGIC) { free(buf); ESP_LOGE(TAG, "bad magic"); return ESP_FAIL; }
    uint32_t ver = rd_u32(buf + 4), n = rd_u32(buf + 8);
    pkg->buf = buf; pkg->buf_size = sz;

    size_t off = 16;
    for (uint32_t i = 0; i < n; ++i) {
        if (off + TOC_ENTRY > (size_t)sz) break;
        const uint8_t *e = buf + off;
        char name[17]; memcpy(name, e, 16); name[16] = 0;
        uint32_t ndim = rd_u32(e + 20);
        uint32_t s0 = rd_u32(e + 24), s1 = rd_u32(e + 28);
        uint32_t doff = rd_u32(e + 40), dnb = rd_u32(e + 44);
        if (doff + dnb > (size_t)sz) { ESP_LOGW(TAG, "section %s OOB", name); off += TOC_ENTRY; continue; }
        const uint8_t *d = buf + doff;
        if (!strcmp(name, "config"))           { pkg->config = (const int32_t *)d; pkg->config_n = (int)s0; }
        else if (!strcmp(name, "weights"))     { pkg->weights = d; pkg->weights_size = dnb; }
        else if (!strcmp(name, "weights_i8"))  { pkg->weights_i8 = d; pkg->weights_i8_size = dnb; }
        else if (!strcmp(name, "mel_mean"))    { pkg->mel_mean = (const float *)d; pkg->mel_n = (int)s0; }
        else if (!strcmp(name, "mel_std"))     { pkg->mel_std = (const float *)d; }
        else if (!strcmp(name, "emb"))         { pkg->emb = (const float *)d; pkg->num_inst = (int)s0; pkg->emb_dim = (int)(ndim >= 2 ? s1 : 0); }
        else if (!strcmp(name, "names"))       { pkg->names = (const char *)d; pkg->names_len = (int)dnb; }
        else if (!strcmp(name, "graph"))       { pkg->graph_ir = d; pkg->graph_ir_size = dnb; }
        off += TOC_ENTRY;
    }
    if (!pkg->weights || !pkg->mel_mean || !pkg->mel_std || !pkg->emb) {
        ESP_LOGE(TAG, "missing required section"); model_store_free(pkg); return ESP_FAIL;
    }
    ESP_LOGI(TAG, "loaded %s ver=%" PRIu32 ": weights=%zuB i8=%zuB N=%d E=%d mel=%d",
             path, ver, pkg->weights_size, pkg->weights_i8_size,
             pkg->num_inst, pkg->emb_dim, pkg->mel_n);
    return ESP_OK;
}

void model_store_free(model_pkg_t *pkg)
{
    if (pkg && pkg->buf) { free(pkg->buf); pkg->buf = NULL; }
}

int model_store_instrument_id(const model_pkg_t *pkg, const char *name)
{
    if (!pkg || !pkg->names || !name) return -1;
    int id = 0;
    const char *p = pkg->names;
    const char *end = pkg->names + pkg->names_len;
    char cur[32];
    while (p < end) {
        int k = 0;
        while (p < end && *p != '\n' && *p != '\0' && k < (int)sizeof(cur) - 1) cur[k++] = *p++;
        cur[k] = 0;
        while (p < end && (*p == '\n' || *p == '\0')) p++;
        if (k > 0) {
            if (strcmp(cur, name) == 0) return id;
            id++;
        }
    }
    return -1;
}
