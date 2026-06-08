#ifndef BNN_MEM_H
#define BNN_MEM_H

#include <stddef.h>

/*
 * 统一内存分配接口
 * - 默认走 malloc / free
 * - 定义 BNN_USE_STATIC_MEM 则使用内置静态堆 (适用 MCU)
 *   静态堆大小由 BNN_STATIC_HEAP_SIZE 控制 (字节)
 * - 用户也可通过 bnn_mem_set_allocator 替换为自己平台的分配器
 */

typedef void *(*bnn_alloc_fn)(size_t size);
typedef void  (*bnn_free_fn)(void *p);

void  bnn_mem_set_allocator(bnn_alloc_fn af, bnn_free_fn ff);

void *bnn_malloc(size_t size);
void *bnn_calloc(size_t n, size_t size);
void  bnn_free(void *p);

/* 内存使用统计 (字节) */
size_t bnn_mem_used(void);
size_t bnn_mem_peak(void);

/* MCU: 尝试把 buf 复制到内部 SRAM 并释放原块; 失败则原样返回 */
void *bnn_try_promote_internal(void *buf, size_t nbytes);

#endif
