#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

struct bnn_workspace {
    unsigned char *buf;
    size_t cap;
    size_t used;
    size_t peak;
    int fixed;   /* 1 = 外部持有 buf, 不增长也不 free */
};

static bnn_workspace_t g_default = { 0 };

/* 16 字节对齐: ESP32-P4 PIE 128-bit 向量指令 (dspm_mult_f32_arp4 等) 要求操作数
 * 至少 16 字节对齐. workspace buf 基址由 bnn_psram_alloc 保证 16 字节对齐,
 * 此宏确保每次 bnn_ws_alloc 后 used 仍保持 16 字节对齐偏移. */
#define WS_ALIGN(x) (((x) + 15u) & ~(size_t)15u)

bnn_workspace_t *bnn_workspace_default(void) { return &g_default; }

bnn_workspace_t *bnn_workspace_create(size_t initial_bytes) {
    bnn_workspace_t *ws = (bnn_workspace_t *)bnn_calloc(1, sizeof(*ws));
    if (!ws) return NULL;
    if (initial_bytes > 0) {
        ws->buf = (unsigned char *)bnn_malloc(initial_bytes);
        if (!ws->buf) { bnn_free(ws); return NULL; }
        ws->cap = initial_bytes;
    }
    return ws;
}

void bnn_workspace_destroy(bnn_workspace_t *ws) {
    if (!ws) return;
    if (!ws->fixed && ws->buf) bnn_free(ws->buf);
    ws->buf = NULL; ws->cap = ws->used = ws->peak = 0; ws->fixed = 0;
    if (ws == &g_default) return;
    bnn_free(ws);
}

static int ws_grow(bnn_workspace_t *ws, size_t need) {
    if (ws->fixed) {
        BNN_LOGE("workspace OOM (固定 buf, need=%zu cap=%zu, 请加大 BNN_WS_SRAM_SIZE)",
                 need, ws->cap);
        return -1;
    }
    size_t cap = ws->cap ? ws->cap : 4096;
    while (cap < need) cap *= 2;
    unsigned char *nb = (unsigned char *)bnn_malloc(cap);
    if (!nb) { BNN_LOGE("workspace grow fail need=%zu", need); return -1; }
    if (ws->buf) {
        if (ws->used) memcpy(nb, ws->buf, ws->used);
        bnn_free(ws->buf);
    }
    ws->buf = nb;
    ws->cap = cap;
    return 0;
}

void *bnn_ws_alloc(bnn_workspace_t *ws, size_t bytes) {
    if (!ws) ws = &g_default;
    bytes = WS_ALIGN(bytes);
    if (ws->used + bytes > ws->cap) {
        if (ws_grow(ws, ws->used + bytes) != 0) return NULL;
    }
    void *p = ws->buf + ws->used;
    ws->used += bytes;
    if (ws->used > ws->peak) ws->peak = ws->used;
    return p;
}

size_t bnn_ws_mark(bnn_workspace_t *ws) { if (!ws) ws = &g_default; return ws->used; }
void   bnn_ws_reset_to(bnn_workspace_t *ws, size_t mark) { if (!ws) ws = &g_default; ws->used = mark; }
void   bnn_ws_reset(bnn_workspace_t *ws) { if (!ws) ws = &g_default; ws->used = 0; }
size_t bnn_ws_used(bnn_workspace_t *ws) { if (!ws) ws = &g_default; return ws->used; }
size_t bnn_ws_peak(bnn_workspace_t *ws) { if (!ws) ws = &g_default; return ws->peak; }

/*
 * 将 workspace 绑定到外部提供的缓冲区 (非 bnn_malloc 分配).
 * 典型用法: MCU 初始化时从内部 SRAM 分配固定块, 避免 workspace 落入慢速 PSRAM.
 *
 *   void *buf = heap_caps_aligned_alloc(16, size, MALLOC_CAP_INTERNAL);
 *   bnn_ws_assign_buf(bnn_workspace_default(), buf, size);
 *
 * 设置后 workspace 不再自动扩容 (grow 返回 -1 并打日志).
 * buf 的生命周期须覆盖 workspace 全部使用期间.
 */
void bnn_ws_assign_buf(bnn_workspace_t *ws, void *buf, size_t cap)
{
    if (!ws) ws = &g_default;
    if (!ws->fixed && ws->buf) bnn_free(ws->buf); /* 释放旧的动态 buf */
    ws->buf   = (unsigned char *)buf;
    ws->cap   = cap;
    ws->used  = 0;
    ws->peak  = 0;
    ws->fixed = 1;
}
