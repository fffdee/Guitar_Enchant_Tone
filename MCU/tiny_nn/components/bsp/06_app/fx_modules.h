/* 电吉他效果器模块接口。
 *
 * 每个效果器实现统一的 fx_node_t 接口, 由 fx_chain 串联调用。
 * 效果器以 in-place 方式处理音频 (输入输出同缓冲), 便于实时流式处理。
 *
 * 已实现效果器:
 *   FX_DISTORTION  失真 (软削波/硬削波/法兹)
 *   FX_EQ          三段均衡 (低/中/高)
 *   FX_DELAY       延迟 (带反馈)
 *   FX_REVERB      混响 (Schroeder: 4 comb + 2 allpass)
 *   FX_COMPRESSOR  压缩器 (动态范围)
 *   FX_NOISE_GATE  噪声门
 *
 * 与音色转换 (i2s_xform) 并列, 共用 I2S 驱动, 但不依赖神经网络模型。
 */
#ifndef FX_MODULES_H
#define FX_MODULES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 效果器类型 */
typedef enum {
    FX_NONE = 0,
    FX_DISTORTION,
    FX_EQ,
    FX_DELAY,
    FX_REVERB,
    FX_COMPRESSOR,
    FX_NOISE_GATE,
    FX_TYPE_MAX
} fx_type_t;

/* 单个参数描述 */
typedef struct {
    char  name[16];   /* 参数名 (drive/level/time_ms...) */
    float value;      /* 当前值 */
    float min_val;    /* 最小值 */
    float max_val;    /* 最大值 */
    float def_val;    /* 默认值 */
} fx_param_t;

#define FX_MAX_PARAMS  8   /* 每个效果器最多参数数 */

/* 效果器节点 - 通用接口 */
typedef struct fx_node {
    fx_type_t   type;
    int         enabled;       /* 1=启用, 0=bypass */
    int         sample_rate;
    fx_param_t  params[FX_MAX_PARAMS];
    int         num_params;
    void       *state;         /* 模块私有状态 (延迟缓冲等) */
    /* in-place 处理: 对 samples[0..n-1] 做处理 */
    void (*process)(struct fx_node *node, float *samples, int n);
    /* 重置内部状态 (清缓冲/重置包络) */
    void (*reset)(struct fx_node *node);
    /* 释放私有状态 */
    void (*deinit)(struct fx_node *node);
} fx_node_t;

/* 创建效果器节点 (按默认参数初始化)。
 * 返回 NULL 表示未知类型或内存不足。
 * 节点用完后由 fx_node_destroy() 释放。 */
fx_node_t *fx_node_create(fx_type_t type, int sample_rate);

/* 销毁节点 (释放私有状态与节点本身) */
void fx_node_destroy(fx_node_t *node);

/* 按名称查找参数索引, 找不到返回 -1 */
int fx_node_param_index(const fx_node_t *node, const char *name);

/* 设置参数值 (自动钳位到 [min, max]), 越界返回 -1 */
int fx_node_set_param(fx_node_t *node, int idx, float value);

/* 按名称设置参数值 */
int fx_node_set_param_by_name(fx_node_t *node, const char *name, float value);

/* 获取效果器类型名称 (静态字符串) */
const char *fx_type_name(fx_type_t type);

/* 按名称解析效果器类型 */
fx_type_t fx_type_from_name(const char *name);

/* 列出所有可用效果器类型名 (用于 help) */
void fx_list_types(char *buf, size_t n);

#ifdef __cplusplus
}
#endif
#endif
