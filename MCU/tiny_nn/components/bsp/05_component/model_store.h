/* 统一模型产物加载: 解析 xform_model.bin (与 PC 端 mask/export.export_model_package 对应).
 * 容器: 头(16B: magic 'XFRM'/ver/n_sections/0) + TOC(每段48B) + 数据段.
 * 段: config(i32) / weights(BNNW raw) / mel_mean(f32) / mel_std(f32) / emb(f32[N,E]) / names. */
#ifndef MODEL_STORE_H
#define MODEL_STORE_H

#include "esp_err.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_PKG_MAGIC 0x4D524658u   /* 'XFRM' */

typedef struct {
    uint8_t       *buf;          /* 整文件缓冲 (本结构拥有) */
    size_t         buf_size;
    const int32_t *config; int config_n;
    const void    *weights; size_t weights_size;   /* BNNW 字节流 */
    const float   *mel_mean; int mel_n;
    const float   *mel_std;
    const float   *emb; int num_inst; int emb_dim;
    const char    *names; int names_len;
    const void    *graph_ir; size_t graph_ir_size;   /* 可选: 图 IR 段 (数据驱动建图), 旧包为 NULL */
} model_pkg_t;

/* 读取并解析模型包. 成功返回 ESP_OK, pkg 内指针指向 pkg->buf. */
esp_err_t model_store_load(const char *path, model_pkg_t *pkg);
void      model_store_free(model_pkg_t *pkg);

/* 乐器名 -> id (按 names 中 \n 分隔的顺序). 未找到返回 -1. */
int model_store_instrument_id(const model_pkg_t *pkg, const char *name);

#ifdef __cplusplus
}
#endif
#endif
