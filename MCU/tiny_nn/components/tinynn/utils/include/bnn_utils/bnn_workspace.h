#ifndef BNN_WORKSPACE_H
#define BNN_WORKSPACE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Workspace (bump arena):
 *  - 训练/推理中, layer 的临时缓冲 (im2col、转置、grad backup 等)
 *    从全局 workspace 一次性 alloc, frame 结束统一 reset.
 *  - 避免反复 malloc/free, 显著降低分片与 latency, 对 MCU 友好.
 *  - 容量超出时自动扩容 (动态后端) 或返回 NULL (静态后端).
 */

typedef struct bnn_workspace bnn_workspace_t;

/* 取全局默认 workspace */
bnn_workspace_t *bnn_workspace_default(void);

/* 创建/销毁自定义 workspace */
bnn_workspace_t *bnn_workspace_create(size_t initial_bytes);
void             bnn_workspace_destroy(bnn_workspace_t *ws);

/* 分配 (按 8 字节对齐). NULL 时使用 default. */
void *bnn_ws_alloc(bnn_workspace_t *ws, size_t bytes);

/* 标记/回退: 嵌套场景下临时使用 */
size_t bnn_ws_mark(bnn_workspace_t *ws);
void   bnn_ws_reset_to(bnn_workspace_t *ws, size_t mark);

/* 完全清空 (一般一个训练 step 结束后调用) */
void   bnn_ws_reset(bnn_workspace_t *ws);

/* 当前使用 / 峰值 (字节) */
size_t bnn_ws_used(bnn_workspace_t *ws);
size_t bnn_ws_peak(bnn_workspace_t *ws);

#ifdef __cplusplus
}
#endif
#endif
