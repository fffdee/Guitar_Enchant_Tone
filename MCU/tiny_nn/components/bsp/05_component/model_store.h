/* 统一模型产物加载: 解析 xform_model.bin (与 PC 端 mask/export.export_model_package 对应).
 * 容器: 头(16B: magic 'XFRM'/ver/n_sections/0) + TOC(每段48B) + 数据段.
 * 段: config(i32) / weights(BNNW raw) / weights_i8(INT8量化) /
 *     mel_mean(f32) / mel_std(f32) / emb(f32[N,E]) / names / graph(IR). */
#ifndef MODEL_STORE_H
#define MODEL_STORE_H

#include "esp_err.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_PKG_MAGIC    0x4D524658u   /* 'XFRM' */
#define BNNW_I8_MAGIC      0x3849574Eu   /* 'NWI8' — INT8 权重段内部魔数 */
#define BNNW_I8_VERSION    1u

typedef struct {
    uint8_t       *buf;          /* 整文件缓冲 (本结构拥有) */
    size_t         buf_size;
    const int32_t *config; int config_n;
    const void    *weights; size_t weights_size;       /* BNNW F32 字节流 */
    const void    *weights_i8; size_t weights_i8_size; /* INT8 量化权重段 (可选, 旧包为 NULL) */
    const float   *mel_mean; int mel_n;
    const float   *mel_std;
    const float   *emb; int num_inst; int emb_dim;
    const char    *names; int names_len;
    const void    *graph_ir; size_t graph_ir_size;     /* 可选: 图 IR 段 (数据驱动建图) */
} model_pkg_t;

/*
 * INT8 权重段迭代器: 按顺序解析 weights_i8 中各 Conv1d 层块.
 *
 * 用法:
 *   model_i8_iter_t it;
 *   model_i8_iter_init(&it, pkg);
 *   while (model_i8_iter_next(&it)) {
 *       // it.W_i8, it.scale, it.Cout, it.Cin_K 已填充
 *   }
 */
typedef struct {
    const uint8_t *p;     /* 当前读取位置 */
    const uint8_t *end;   /* 段尾 */
    uint32_t       left;  /* 剩余层数 */
    /* 当前层输出 */
    const int8_t  *W_i8;  /* [Cout, Cin_K] INT8 权重 */
    const float   *scale; /* [Cout] per-channel 量化尺度 */
    uint32_t       Cout;
    uint32_t       Cin_K;
} model_i8_iter_t;

/* 初始化迭代器 (若 pkg->weights_i8 为 NULL 则 left=0). */
void model_i8_iter_init(model_i8_iter_t *it, const model_pkg_t *pkg);

/* 读取下一层, 返回 1 成功 / 0 无更多层. */
int  model_i8_iter_next(model_i8_iter_t *it);

/* 读取并解析模型包. 成功返回 ESP_OK, pkg 内指针指向 pkg->buf. */
esp_err_t model_store_load(const char *path, model_pkg_t *pkg);
void      model_store_free(model_pkg_t *pkg);

/* 乐器名 -> id (按 names 中 \n 分隔的顺序). 未找到返回 -1. */
int model_store_instrument_id(const model_pkg_t *pkg, const char *name);

#ifdef __cplusplus
}
#endif
#endif
