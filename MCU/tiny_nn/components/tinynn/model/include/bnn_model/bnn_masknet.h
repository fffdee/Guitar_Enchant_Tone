#ifndef BNN_MASKNET_H
#define BNN_MASKNET_H

#include "bnn_frontend/bnn_mask_config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 复音频域谱掩码 端到端推理运行时 (IMPLEMENTATION.md 主线) —— "项目 = 框架组件组合".
 *
 * 组合:
 *   specfront (frontend) : 帧 -> 对数梅尔(标准化) + 幅度/相位
 *   graph     (graph)    : MaskNet 主干 conv1d+FiLM, 三头(幅度掩码/相位残差/噪声带)
 *   specsynth (synth)    : 掩码作用+相位复用残差+ISTFT/OLA+噪声带 -> 音频
 *
 * 原生复音(对幅度谱施加非负增益, 保留所有音高), 切换乐器=换嵌入(set_instrument).
 * 计算图帧数 T 创建时固定; process_audio 内部 center 分帧 + 分块, 流式 OLA 输出.
 */
typedef struct bnn_masknet bnn_masknet_t;

bnn_masknet_t *bnn_masknet_create(const bnn_mask_cfg_t *cfg, int num_instruments, int block_frames);

/* 数据驱动建图: 若 ir_buf 非空, 用图 IR (bnn_graph_build_from_ir) 构建主干 CNN
 * (改网络结构=换数据, 不改固件); ir_buf 为空时回退到内置硬编码 build_graph.
 * IR 输出节点约定: [0]=幅度掩码头, [1]=相位残差头, [2]=噪声带头. */
bnn_masknet_t *bnn_masknet_create_ir(const bnn_mask_cfg_t *cfg, int num_instruments,
                                     int block_frames, const void *ir_buf, size_t ir_len);
void           bnn_masknet_destroy(bnn_masknet_t *m);

/* 加载网络权重 (BNNW; 顺序由 Python export_masknet_weights 保证). 返回 0 成功. */
int  bnn_masknet_load_weights_mem(bnn_masknet_t *m, const void *buf, size_t nbytes);

/* 梅尔标准化 mean/std (各 n_mels). */
void bnn_masknet_set_mel_norm(bnn_masknet_t *m, const float *mean, const float *std);

/* 乐器嵌入表 [n][dim], dim 须 = emb_dim. */
int  bnn_masknet_set_embedding_table(bnn_masknet_t *m, const float *table, int n, int dim);
int  bnn_masknet_set_instrument(bnn_masknet_t *m, int instrument_id);
int  bnn_masknet_set_embedding(bnn_masknet_t *m, const float *emb, int dim);

/* 噪声开关 / 增益平滑 / 噪声门(dB) / 复位流式状态 */
void bnn_masknet_set_add_noise(bnn_masknet_t *m, int on);
void bnn_masknet_set_smooth(bnn_masknet_t *m, float a);
void bnn_masknet_set_noise_gate(bnn_masknet_t *m, float db);
void bnn_masknet_reset(bnn_masknet_t *m);

/*
 * 处理一段连续音频 (center 分帧):
 *   audio[n_audio] -> out_audio (长度 >= n_audio); out_n 写出样点数 (可 NULL). 返回 0 成功.
 */
int  bnn_masknet_process_audio(bnn_masknet_t *m, const float *audio, int n_audio,
                               float *out_audio, int *out_n);

size_t bnn_masknet_num_params(const bnn_masknet_t *m);
int    bnn_masknet_block_frames(const bnn_masknet_t *m);

#ifdef __cplusplus
}
#endif
#endif
