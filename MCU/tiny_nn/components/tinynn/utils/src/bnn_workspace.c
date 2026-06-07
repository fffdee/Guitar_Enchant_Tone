#include "bnn_utils/bnn_workspace.h"
#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <string.h>

struct bnn_workspace {
    unsigned char *buf;
    size_t cap;
    size_t used;
    size_t peak;
};

static bnn_workspace_t g_default = { 0 };

#define WS_ALIGN(x) (((x) + 7u) & ~(size_t)7u)

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
    if (ws == &g_default) {
        if (ws->buf) bnn_free(ws->buf);
        ws->buf = NULL; ws->cap = ws->used = ws->peak = 0;
        return;
    }
    if (ws->buf) bnn_free(ws->buf);
    bnn_free(ws);
}

static int ws_grow(bnn_workspace_t *ws, size_t need) {
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
