#include "bnn_utils/bnn_mem.h"
#include "bnn_utils/bnn_log.h"
#include <stdlib.h>
#include <string.h>

#ifndef BNN_STATIC_HEAP_SIZE
#define BNN_STATIC_HEAP_SIZE (64 * 1024)
#endif

static size_t g_used = 0;
static size_t g_peak = 0;

#ifdef BNN_USE_STATIC_MEM
/*
 * 简易首次适配空闲链表 (single-list first-fit).
 * 嵌入式资源紧张, 不做复杂合并 (仅相邻空闲合并).
 */
typedef struct blk {
    size_t      size;     /* 用户区大小 */
    int         used;
    struct blk *next;
} blk_t;

static unsigned char g_heap[BNN_STATIC_HEAP_SIZE] __attribute__((aligned(8)));
static blk_t *g_head = NULL;

static void static_init(void) {
    if (g_head) return;
    g_head = (blk_t *)g_heap;
    g_head->size = BNN_STATIC_HEAP_SIZE - sizeof(blk_t);
    g_head->used = 0;
    g_head->next = NULL;
}

static void *static_alloc(size_t sz) {
    static_init();
    sz = (sz + 7u) & ~(size_t)7u;
    blk_t *b = g_head;
    while (b) {
        if (!b->used && b->size >= sz) {
            /* 分裂 */
            if (b->size >= sz + sizeof(blk_t) + 16) {
                blk_t *n = (blk_t *)((unsigned char *)(b + 1) + sz);
                n->size = b->size - sz - sizeof(blk_t);
                n->used = 0;
                n->next = b->next;
                b->next = n;
                b->size = sz;
            }
            b->used = 1;
            g_used += b->size;
            if (g_used > g_peak) g_peak = g_used;
            return (void *)(b + 1);
        }
        b = b->next;
    }
    BNN_LOGE("static heap OOM req=%zu used=%zu", sz, g_used);
    return NULL;
}

static void static_free(void *p) {
    if (!p) return;
    blk_t *b = (blk_t *)p - 1;
    b->used = 0;
    g_used -= b->size;
    /* 合并相邻空闲块 */
    blk_t *cur = g_head;
    while (cur && cur->next) {
        if (!cur->used && !cur->next->used) {
            cur->size += sizeof(blk_t) + cur->next->size;
            cur->next  = cur->next->next;
        } else {
            cur = cur->next;
        }
    }
}

static bnn_alloc_fn g_alloc = static_alloc;
static bnn_free_fn  g_free  = static_free;

#else  /* 动态堆 */

typedef struct {
    size_t size;
} hdr_t;

static void *dyn_alloc(size_t sz) {
    hdr_t *h = (hdr_t *)malloc(sizeof(hdr_t) + sz);
    if (!h) { BNN_LOGE("malloc fail sz=%zu", sz); return NULL; }
    h->size = sz;
    g_used += sz;
    if (g_used > g_peak) g_peak = g_used;
    return (void *)(h + 1);
}
static void dyn_free(void *p) {
    if (!p) return;
    hdr_t *h = (hdr_t *)p - 1;
    g_used -= h->size;
    free(h);
}

static bnn_alloc_fn g_alloc = dyn_alloc;
static bnn_free_fn  g_free  = dyn_free;
#endif

void bnn_mem_set_allocator(bnn_alloc_fn af, bnn_free_fn ff) {
    if (af) g_alloc = af;
    if (ff) g_free  = ff;
}

void *bnn_malloc(size_t size)            { return g_alloc(size); }
void  bnn_free(void *p)                  { g_free(p); }
void *bnn_calloc(size_t n, size_t size)  {
    size_t total = n * size;
    void *p = g_alloc(total);
    if (p) memset(p, 0, total);
    return p;
}
size_t bnn_mem_used(void) { return g_used; }
size_t bnn_mem_peak(void) { return g_peak; }
