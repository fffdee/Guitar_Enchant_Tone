/*
 * bnn_nn.h — NN 加速算子后端虚表 (架构与 bnn_dsp_backend_t 相同).
 *
 * 设计目的:
 *   tinynn 层 (layer_conv1d 等) 通过此虚表调用加速实现,
 *   而不直接依赖 ESP-NN 头文件 (ESP-NN 在 bsp 组件中).
 *   这样 tinynn 组件与硬件无关, 保持可移植性.
 *
 * 注册方: bsp/05_component/nn_accel.c 在 nn_accel_init_int8() 中
 *         调用 bnn_nn_set_backend(nn_accel_nn_backend()).
 *
 * 使用方: tinynn/layer/src/layer_conv1d.c 在 BNN_CONV1D_INT8_ACCEL
 *         编译宏开启时, 调用 bnn_nn_get_backend()->conv1d_s8(...).
 */
#ifndef BNN_NN_H
#define BNN_NN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * INT8 Conv1D 函数指针 — Conv2D(H=1) 形式调用 ESP-NN.
 *
 *  input       [Cin, T]          int8  (row-major, no batch)
 *  in_offset   0 (对称量化 zero_point)
 *  T           时间长度 (input_wd)
 *  Cin         输入通道
 *  pad         时间轴 same-padding 长度 (pad_wd)
 *  stride      步长 (stride_wd)
 *  dilation    膨胀率 (dilation_wd)
 *  filter      [Cout, Cin, K]    int8
 *  filter_offset  0 (对称量化)
 *  Cout        输出通道
 *  K           卷积核宽度
 *  bias        [Cout]            int32 (= 0 向量, 若不含偏置可为全零)
 *  output      [Cout, To]        int8
 *  out_offset  0 (对称量化)
 *  out_shift   [Cout]            per-channel 右移 (定点反量化)
 *  out_mult    [Cout]            per-channel 乘子 (定点反量化), 可为 NULL(全 1)
 *  To          输出时间长度
 */
typedef void (*bnn_nn_conv1d_s8_fn)(
    const int8_t  *input,   int32_t in_offset,
    int T, int Cin, int pad, int stride, int dilation,
    const int8_t  *filter,  int32_t filter_offset,
    int Cout, int K,
    const int32_t *bias,
    int8_t  *output,        int32_t out_offset,
    const int32_t *out_shift, const int32_t *out_mult,
    int To
);

typedef struct bnn_nn_backend {
    const char          *name;
    bnn_nn_conv1d_s8_fn  conv1d_s8;   /* NULL = 不可用 */
} bnn_nn_backend_t;

void                    bnn_nn_set_backend(const bnn_nn_backend_t *be);
const bnn_nn_backend_t *bnn_nn_get_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* BNN_NN_H */
