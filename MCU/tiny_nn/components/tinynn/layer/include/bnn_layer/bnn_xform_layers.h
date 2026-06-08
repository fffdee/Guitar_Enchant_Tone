#ifndef BNN_XFORM_LAYERS_H
#define BNN_XFORM_LAYERS_H

/*
 * ESP-GT-XFORM 推理层的"层私有配置"(经 bnn_layer_cfg_t.extra 传入).
 * 这是对框架灵活性改进(原则3)的示范: 通用 cfg 装不下的参数走 extra 强类型结构.
 *
 * 这些层均为"推理前向"实现(训练在 PyTorch 完成), backward 置 NULL, graph 自动跳过.
 */

#include "bnn_layer.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* conv1d: 1D 膨胀卷积. 通用字段 in_channels/out_channels/kernel/stride/padding/dilation 已够用,
 * 此结构仅作可选覆盖位 (一般传 NULL). */
typedef struct bnn_conv1d_cfg {
    int reserved;
} bnn_conv1d_cfg_t;

/* film: 条件调制. 由嵌入 e 经线性层生成逐通道 (gamma,beta).
 *   channels       = 被调制特征图的通道数 C
 *   embedding_dim  = 条件向量维度 E
 *   gamma_plus_one = 1: y=(1+gamma)*x+beta (DDSP/ConditionalDDSPNet 约定, 零初始即恒等)
 *                    0: y=gamma*x+beta     (MaskNet/§3.4 约定)
 * 亦可用通用字段: out_channels=C, in_features=E (extra 为 NULL 时回退读取, gamma_plus_one 默认 1). */
typedef struct bnn_film_cfg {
    int channels;
    int embedding_dim;
    int gamma_plus_one;
} bnn_film_cfg_t;

/* embedding: 整数 id -> dim 维向量查表.
 *   num_embeddings = 乐器/类别数 N
 *   dim           = 向量维度 */
typedef struct bnn_embedding_cfg {
    int num_embeddings;
    int dim;
} bnn_embedding_cfg_t;

/*
 * conv1d_set_weights_i8 — 向 conv1d 层注入 INT8 量化权重.
 *
 *  layer  : bnn_layer_t* (type_name 须为 "conv1d")
 *  W_i8   : int8 [Cout, Cin_K] 量化权重 (行主序)
 *  scale  : float [Cout]       per-output-channel 量化尺度
 *  Cout   : 输出通道数 (须与层配置一致)
 *  Cin_K  : Cin * K (须与层配置一致)
 *
 * 返回 0 成功, -1 失败 (类型不匹配或 OOM 或 INT8 路径未编译).
 * 由 bnn_masknet_load_weights_i8_mem 迭代调用.
 */
int conv1d_set_weights_i8(bnn_layer_t *layer,
                          const int8_t *W_i8, const float *scale,
                          int Cout, int Cin_K);

#ifdef __cplusplus
}
#endif
#endif
