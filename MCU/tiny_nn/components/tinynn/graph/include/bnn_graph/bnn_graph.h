#ifndef BNN_GRAPH_H
#define BNN_GRAPH_H

#include "bnn_layer/bnn_layer.h"
#include "bnn_utils/bnn_optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 计算图: 面向用户的接口.
 *
 * 设计:
 *  - 节点列表 + 显式输入依赖 (最多 2 输入), 按添加顺序即拓扑顺序
 *  - 训练时维护每个节点输出的梯度缓冲 (按需懒分配)
 *  - 输入节点: bnn_graph_add_input
 *  - 普通层节点: bnn_graph_add_layer(name, cfg, dep_ids...)
 *  - 前向: bnn_graph_forward(g, input_data...)
 *  - 反向: bnn_graph_backward(g, d_output)
 *  - 与 optimizer 绑定: bnn_graph_collect_params -> add to optimizer
 */

typedef struct bnn_graph bnn_graph_t;

#define BNN_GRAPH_BAD_ID (-1)

bnn_graph_t *bnn_graph_create(void);
void         bnn_graph_destroy(bnn_graph_t *g);

/* 添加输入节点; shape 描述 (batch 可填 -1 在推理前由用户提供) */
int  bnn_graph_add_input(bnn_graph_t *g, int ndim, const int *shape);

/* 添加 layer 节点. dep_ids 为依赖节点 id 数组, n_dep 个 (一般 1, residual=2). 返回节点 id. */
int  bnn_graph_add_layer(bnn_graph_t *g, const char *layer_name,
                         const bnn_layer_cfg_t *cfg,
                         const int *dep_ids, int n_dep);

/* 标记输出节点 */
void bnn_graph_set_output(bnn_graph_t *g, int node_id);

/* 提供输入张量数据 (按输入节点顺序). x_data[i] 大小 = 对应 input 节点 numel*sizeof(float) */
/* 同时可指定动态 batch (>0 时覆盖 shape[0]) */
int  bnn_graph_feed_input(bnn_graph_t *g, int input_index, const float *data, int batch);

/* 前向; 返回输出张量 (graph 拥有, 用户只读) */
bnn_tensor_t *bnn_graph_forward(bnn_graph_t *g);

/*
 * 反向:
 *  dL_dout: 大小 = output numel; 由 loss 函数提供
 *  会累积参数梯度 (用户在每个 step 前需 zero_grad)
 */
void bnn_graph_backward(bnn_graph_t *g, const float *dL_dout);

/* 将图中所有 layer 的可训练参数注册到 optimizer */
void bnn_graph_collect_params(bnn_graph_t *g, bnn_optimizer_t *opt);

/* 调试 / 信息 */
int  bnn_graph_node_count(const bnn_graph_t *g);

/* 取某节点前向输出张量 (多头网络: 一次 forward 后按节点 id 读取各头). 失败返回 NULL. */
bnn_tensor_t *bnn_graph_node_output(const bnn_graph_t *g, int node_id);

/* 一次性参数总数 (float 个数) */
size_t bnn_graph_total_params(const bnn_graph_t *g);

/* 保存 / 加载所有可训练参数 (按图拓扑顺序、layer 内部顺序). 简易二进制:
 *   magic(4)='BNNW' | version(u32)=1 | num_params(u64) | floats... */
int bnn_graph_save_weights(const bnn_graph_t *g, const char *path);
int bnn_graph_load_weights(const bnn_graph_t *g, const char *path);

/* 从内存字节流加载权重 (与 save_weights 同格式), 适用无文件系统的 MCU:
 * 把导出的 .bin 以 C 数组 #include 进固件, 直接传入即可. 返回 0 成功. */
int bnn_graph_load_weights_mem(const bnn_graph_t *g, const void *buf, size_t nbytes);

/* 切换图中所有支持 train/infer 模式的层 (BN/Dropout). 通过重建该层完成; 推荐做法:
 * 用户在构图时直接根据需要传入 cfg.activation 字段. 这里只提供查询.
 */
int bnn_graph_num_layers(const bnn_graph_t *g);

#ifdef __cplusplus
}
#endif
#endif
