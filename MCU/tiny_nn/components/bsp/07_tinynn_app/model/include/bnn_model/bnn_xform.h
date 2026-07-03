#ifndef BNN_XFORM_H
#define BNN_XFORM_H

#include "bnn_frontend/bnn_xform_config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ESP-GT-XFORM 端到端推理运行时 —— "项目 = 框架组件的组合" 的集中体现.
 *
 * 内部串起三大框架组件:
 *   frontend (bnn_frontend)  : 音频帧 -> 20 维吉他特征 + f0
 *   graph    (bnn_graph)     : 按 ConditionalDDSPNet 结构构建的条件 DDSP 网络
 *                              (conv1d / film / 由 host 做嵌入查表后喂入)
 *   synth    (bnn_synth)     : 40 维参数 + f0 -> 音频 (谐波 + 子带噪声)
 *
 * 切换乐器零成本: 只更换 8 维嵌入向量 (set_instrument / set_embedding).
 *
 * 设计取舍:
 *   - 计算图输入帧数 T 在创建时固定 (block_frames), 适配 conv1d 的时序膨胀感受野;
 *     不足一块时内部补零, 只输出有效音频长度.
 *   - 嵌入查表放在 host 侧 (而非图内 embedding 层), 因为嵌入既要拼进卷积输入,
 *     又要喂给 FiLM 第二输入; host 查表后同时供给两处, 接线最简。
 */
typedef struct bnn_xform bnn_xform_t;

/* 网络结构描述 (默认与 configs/train_ddsp.yaml 一致) */
typedef struct bnn_xform_net_cfg {
    int kernel;          /* 卷积核 (默认 3) */
    int n_conv;          /* 卷积层数 (默认 4) */
    int channels[8];     /* 各卷积输出通道 (默认 32,64,128,64) */
    int dilations[8];    /* 各卷积膨胀率 (默认 1,2,4,1) */
    int film_mask;       /* bit i=1: 第 i 卷积(0-based)后接 FiLM (默认 (1<<1)|(1<<2)) */
    int embedding_dim;   /* 乐器嵌入维度 (默认 8) */
    int num_instruments; /* 乐器数 (用于嵌入表行数; 默认 4) */
} bnn_xform_net_cfg_t;

void bnn_xform_net_cfg_default(bnn_xform_net_cfg_t *net);

/* 创建运行时. cfg 提供音频/特征/DDSP 维度; net 为网络结构 (NULL 用默认); block_frames 为每块帧数 T. */
bnn_xform_t *bnn_xform_create(const bnn_xform_cfg_t *cfg,
                              const bnn_xform_net_cfg_t *net,
                              int block_frames);
void         bnn_xform_destroy(bnn_xform_t *m);

/* 加载网络权重 (BNNW 内存格式, 顺序由 Python 导出器 export_bnn_weights 保证). 返回 0 成功. */
int bnn_xform_load_weights_mem(bnn_xform_t *m, const void *buf, size_t nbytes);

/* 设定特征标准化 mean/std (各 feature_dim 长). 传 NULL 关闭标准化. */
void bnn_xform_set_feature_norm(bnn_xform_t *m, const float *mean, const float *std);

/* 设定乐器嵌入表 [n][dim]; dim 须等于 embedding_dim. */
int  bnn_xform_set_embedding_table(bnn_xform_t *m, const float *table, int n, int dim);

/* 选择乐器 (从嵌入表取行) 或直接给定嵌入向量. 下次 process 生效. */
int  bnn_xform_set_instrument(bnn_xform_t *m, int instrument_id);
int  bnn_xform_set_embedding(bnn_xform_t *m, const float *emb, int dim);

/*
 * 处理一块已分帧音频:
 *   frames    : [n_frames][frame_size] 连续分析帧 (调用方按 hop 推进分帧)
 *   n_frames  : 1..block_frames (不足按补零处理)
 *   out_audio : 输出, 长度 >= n_frames*hop_size
 * 返回 0 成功.
 */
int bnn_xform_process_frames(bnn_xform_t *m, const float *frames, int n_frames,
                             float *out_audio);

/*
 * 便捷: 直接处理一段连续音频, 内部按 frame_size/hop 分帧并按 block_frames 分块.
 *   audio   : [n_audio]
 *   out_audio: 输出, 长度 >= ((n_audio-frame_size)/hop+1)*hop
 *   out_n   : 写出的样点数 (可为 NULL)
 * 返回 0 成功.
 */
int bnn_xform_process_audio(bnn_xform_t *m, const float *audio, int n_audio,
                            float *out_audio, int *out_n);

/* 复位前端流式状态 (切换音轨/重新开始时调用) */
void bnn_xform_reset(bnn_xform_t *m);

/* 信息查询 */
int    bnn_xform_block_frames(const bnn_xform_t *m);
size_t bnn_xform_num_params(const bnn_xform_t *m);

#ifdef __cplusplus
}
#endif
#endif
