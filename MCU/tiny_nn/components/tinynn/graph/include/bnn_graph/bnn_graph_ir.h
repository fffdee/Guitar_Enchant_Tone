#ifndef BNN_GRAPH_IR_H
#define BNN_GRAPH_IR_H

#include "bnn_graph/bnn_graph.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 图 IR (Intermediate Representation): 数据驱动地描述一张计算图, 让"改网络结构 = 换数据"
 * 而无需改固件. 这是框架可扩展性的核心: 任意网络只要能表达成 IR, 即可用
 * bnn_graph_build_from_ir 在设备上动态建图.
 *
 * IR 也是 ONNX 的稳定中间接口 (混合路线): PC 端 onnx->IR 转换器产出此格式,
 * 设备端只解析 IR; 将来若要设备端直接读 .onnx, 只需新增 onnx(protobuf)->IR 前端,
 * 复用本加载器, 不动 graph/layer/operator 层.
 *
 * 二进制布局 (小端):
 *   Header 32B: magic('BGIR') | version | n_nodes | n_outputs | weights_off | weights_len | rsv | rsv
 *   Node[n_nodes], 每个 84B (见 bnn_ir_node 字段顺序, 全 4 字节对齐, 无 padding)
 *   Outputs: n_outputs 个 i32 (IR 节点序号), 紧跟节点表之后
 *   Weights: BNNW 字节流, 位于 weights_off (与 bnn_graph_load_weights_mem 同格式)
 *
 * 节点 kind: 0=input(用 ndim+shape), 1=layer(用 type/cfg/deps).
 * film 节点: gamma_plus_one 存在 extra0; channels=out_channels, embedding_dim=in_features.
 */

#define BNN_IR_MAGIC      0x52494742u   /* 'BGIR' 小端 */
#define BNN_IR_VERSION    1u
#define BNN_IR_NODE_BYTES 84
#define BNN_IR_MAX_OUTPUTS 8

typedef struct {
    bnn_graph_t *graph;                 /* 成功时拥有的图; 失败为 NULL */
    int          out_nodes[BNN_IR_MAX_OUTPUTS];  /* 输出节点 id (多头: [0]主输出) */
    int          n_out;
} bnn_ir_model_t;

/* 从 IR 字节流构建计算图并加载权重 (若 IR 内含权重段). 返回 0 成功.
 * 失败时 out->graph 为 NULL. 调用方用 bnn_graph_destroy(out->graph) 释放. */
int bnn_graph_build_from_ir(const void *buf, size_t nbytes, bnn_ir_model_t *out);

#ifdef __cplusplus
}
#endif
#endif
