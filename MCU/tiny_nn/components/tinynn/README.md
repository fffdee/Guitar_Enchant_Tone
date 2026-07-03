# tinynn — 纯 C 轻量神经网络推理框架

面向 MCU / Linux 的极简**通用推理引擎**。设计三原则：**与硬件解耦**、**与项目解耦**、**移植方便**。

框架只包含与项目无关的通用能力，项目特有的代码（模型/合成/前端/效果器等）放在 `bsp/07_tinynn_app/` 中。

| 层组 | 目录 | 作用 |
|----|------|------|
| 工具层 | `utils/`    | 内存池(静态/动态) / tensor / workspace(bump) / log(可重定向) / optimizer / loss / dataset |
| 算子层 | `operator/` | compute 后端(`bnn_op`) + DSP/FFT 后端(`bnn_dsp`) + NN 算子(`bnn_nn`) |
| 结构层 | `layer/`    | dense / conv2d / **conv1d** / **film**(可选 γ/(1+γ)) / **embedding** / activation(含 softplus/缩放) / residual…，vtbl + 静态注册 |
| 计算图层 | `graph/`  | 用户构图 / 前向 / 反向 / 多头输出 / 权重存取(文件可裁剪, 内存加载始终可用) / 图 IR(数据驱动建图) |

> 项目特有的扩展（放在 `bsp/07_tinynn_app/`）：
> - **前端组** `frontend/`：DDSP 特征提取(`bnn_frontend`)；谱掩码前端(`bnn_specfront`)
> - **合成组** `synth/`：DDSP 谐波合成(`bnn_synth`)；谱掩码合成(`bnn_specsynth`)
> - **模型层** `model/`：`bnn_xform`(单音 DDSP)；`bnn_masknet`(复音谱掩码)
> - **效果器** `effect/`：`bnn_fx` 传统 DSP 失真
> - **项目算子**：频谱矩阵(`bnn_specmat`)；复音判别(`bnn_polyphony`)

## 三原则的体现

- **与硬件解耦**
  - 计算后端 `bnn_op_set_backend()`、FFT 后端 `bnn_dsp_set_backend()`、内存分配 `bnn_mem_set_allocator()`、计时 `bnn_time_set_source()` 全是可注入的函数指针。
  - 日志经 `bnn_log_set_sink()` 重定向到 UART/RTT；MCU 默认不引用 stdio（`BNN_LOG_USE_STDIO=0`）。
  - 文件 I/O 由 `BNN_ENABLE_FILE_IO` 控制；MCU 默认关闭 `fopen` 等，改用 `bnn_graph_load_weights_mem` 从 Flash 数组加载。
- **移植方便**
  - 根 `CMakeLists.txt` 工具链无关（隔离 GCC/Clang 专属 flag，兼容 MSVC）。
  - `port/esp-idf/CMakeLists.txt` 即拿即用的 ESP-IDF 组件构建文件。
  - 纯 C11，无第三方依赖；推理路径不强制 malloc（workspace bump 分配）。
- **面向对象**
  - `bnn_layer_vtbl_t` 虚表 + 工厂注册；`bnn_op`/`bnn_dsp` 后端虚表；`bnn_frontend`/`bnn_synth`/`bnn_xform` 均为不透明对象（create/destroy/process）。

## 构建 (Linux/host)

```bash
cd MCU/Tinynn
mkdir build && cd build
cmake ..
make -j
./masknet_infer   # 复音谱掩码 端到端推理示例
./xform_infer     # 单音 DDSP 端到端推理示例
./fx_demo         # 失真效果器示例 (§7)
./classify_mlp    # 训练示例
```

## MCU / ESP32 移植

- 推理-only：定义 `BNN_PLATFORM_MCU`（自动令 `BNN_ENABLE_TRAINING=0`、`BNN_ENABLE_FILE_IO=0`、`BNN_LOG_USE_STDIO=0`）。
- 静态堆：`-DBNN_USE_STATIC_MEM=ON -DBNN_STATIC_HEAP_SIZE=65536`。
- ESP-IDF：把 `port/esp-idf/` 作为组件（详见该文件头注释），需要时 `bnn_dsp_set_backend()` 注入 ESP-DSP 的 FFT。
- 不支持 GCC `constructor` 的工具链：`bnn_xform_create()` 内部已兜底 `bnn_layers_init_all()`；裸用 graph 时在 `main` 开头自行调用。
- 日志重定向：`bnn_log_set_sink(my_uart_sink, ctx)`。

## ESP-GT-XFORM：训练 → 部署 权重流程

1. PC 端训练后，导出器（`src/esp_xform/train/export.py`）产出：
   - `xform_weights.bin` / `xform_weights.h`：网络权重，BNNW 格式，**顺序与 C 计算图一致**。
   - `instrument_embeddings.h`、`feature_mean.h`、`feature_std.h`。
2. 固件侧：

```c
#include "bnn_model/bnn_xform.h"
#include "xform_weights.h"          // unsigned char xform_weights_bin[]
#include "instrument_embeddings.h"  // float instrument_embeddings[N][8]
#include "feature_mean.h"
#include "feature_std.h"

bnn_xform_cfg_t cfg; bnn_xform_cfg_default(&cfg);
bnn_xform_t *m = bnn_xform_create(&cfg, NULL, /*block_frames*/32);
bnn_xform_load_weights_mem(m, xform_weights_bin, XFORM_WEIGHTS_BIN_LEN);
bnn_xform_set_feature_norm(m, feature_mean, feature_std);
bnn_xform_set_embedding_table(m, &instrument_embeddings[0][0],
                              INSTRUMENT_EMBEDDINGS_ROWS, INSTRUMENT_EMBEDDINGS_COLS);
bnn_xform_set_instrument(m, 1);                 // 切换乐器=只换嵌入
bnn_xform_process_audio(m, in, n_in, out, &n_out);
```

> 注：网络/谐波路径与 PC 端数值高度一致；子带噪声为统计等价（各带 RMS 受控），因 RNG 实现不同非逐样点一致。

## 复音谱掩码：训练 → 部署 权重流程

1. PC 端 `src/esp_xform/mask/` 训练（`scripts/preprocess_mask.py` → `scripts/train_mask.py`），导出器产出：
   - `masknet_weights.bin` / `.h`（BNNW，顺序与 C 计算图一致）；
   - `mel_inv.h`/`phase_inv.h`/`noise_fb.h`/`mel_basis.h`（C 也可自建，见 `bnn_specmat`）；
   - `mel_mean.h` / `mel_std.h`、`instrument_embeddings.h`。
2. 固件侧：

```c
#include "bnn_model/bnn_masknet.h"
#include "masknet_weights.h"        // unsigned char masknet_weights_bin[]
#include "instrument_embeddings.h"  // float instrument_embeddings[N][16]
#include "mel_mean.h"
#include "mel_std.h"

bnn_mask_cfg_t cfg; bnn_mask_cfg_default(&cfg);
bnn_masknet_t *m = bnn_masknet_create(&cfg, /*num_inst*/N, /*block_frames*/64);
bnn_masknet_load_weights_mem(m, masknet_weights_bin, MASKNET_WEIGHTS_BIN_LEN);
bnn_masknet_set_mel_norm(m, mel_mean, mel_std);
bnn_masknet_set_embedding_table(m, &instrument_embeddings[0][0],
                                INSTRUMENT_EMBEDDINGS_ROWS, INSTRUMENT_EMBEDDINGS_COLS);
bnn_masknet_set_instrument(m, 1);   // 切换乐器=换嵌入
bnn_masknet_process_audio(m, in, n_in, out, &n_out);   // 原生复音
```

> 谱矩阵 `mel_basis/mel_inv/phase_inv/noise_fb` 在 C 端由 `bnn_specmat` 按与 Python 相同公式自建，
> 故固件无需附带这些 `.h`（导出仅为校验/可选）。相位残差头若量化敏感或超算力预算，可关闭(`dphi=0`)。

## 扩展

- 新增层：写一个 `.c`，实现 `bnn_layer_vtbl_t` + `BNN_REGISTER_LAYER(name, &vtbl)`，并在 `bnn_layer.c` 的 `bnn_layers_init_all` 兜底列表登记。
- 新增算子后端：填 `bnn_op_backend_t`，`bnn_op_set_backend(&my_be)`。
- 新增 FFT 后端：填 `bnn_dsp_backend_t`，`bnn_dsp_set_backend(&my_dsp)`。
- 层私有配置：通用 `bnn_layer_cfg_t` 装不下时走 `cfg.extra` 强类型结构（见 `bnn_xform_layers.h`）。

## 图 IR（数据驱动建图 / ONNX 稳定中间接口）

`graph/bnn_graph_ir.{h,c}` 定义了一种紧凑的二进制"图 IR"，描述一张计算图（输入节点 +
层节点 + 依赖 + 多头输出 + 可选内嵌 BNNW 权重）。设备端用 `bnn_graph_build_from_ir()`
按 IR 动态建图——**改网络结构 = 换数据文件，无需改固件**，这是框架可扩展性的核心。

二进制布局（小端，与 `bnn_graph_ir.h` 一致）：

```
Header 32B: magic('BGIR') | version | n_nodes | n_outputs | weights_off | weights_len | rsv | rsv
Node 84B  : type[16] | kind | cfg[9](i32) | param(f32) | ndep_or_ndim | a[4] | extra0
Outputs   : n_outputs 个 i32（IR 节点序号）
Weights   : 可选 BNNW 字节流（magic('BNNW') ver count(u64) f32[]），位于 weights_off
```

- 节点 `kind`：0=input（用 `ndim`+`shape`），1=layer（用 `type`/`cfg`/`deps`）。
- `film` 节点：`gamma_plus_one` 存 `extra0`；`channels=out_channels`，`embedding_dim=in_features`。
- 权重顺序：按节点拓扑序、层内先 W 后 b，与 `bnn_graph` 收集参数顺序一致。

PC 侧写入器：`src/esp_xform/onnx/ir_writer.py`（纯 stdlib，与本 C 格式逐字段对应）。
`bnn_masknet` 已支持数据驱动：模型包 `xform_model.bin` 若含 `graph` 段则用 IR 建图，
否则回退内置 `build_graph`（向后兼容）。

## ONNX（混合路线）

- **PC 转换器（已实现）**：`tools/onnx_to_bnn_ir.py` 把标准 `.onnx` 映射为图 IR(.bin)，
  支持 Conv(1D/2D)/Relu/Sigmoid/Tanh/Softplus/Add(残差)/Gemm/MatMul/Flatten，
  不支持的算子明确报错。产物自包含权重，设备端 `bnn_graph_build_from_ir` 直接运行。
- **设备端 ONNX 前端（预留）**：图 IR 是稳定中间接口。将来若要设备端直接读 `.onnx`，
  只需新增一个 `onnx(protobuf)->IR(内存)` 前端，复用 `bnn_graph_build_from_ir`，
  不动 graph/layer/operator 层。

## 命令行与双核（ESP-GT-XFORM 固件）

- 命令行（`esp_console` REPL，跑在 core0）+ 推理 worker（队列任务，固定 core1）解耦：
  推理进行时命令行仍可用。
- 常用命令：`model`（模型信息）、`infer <乐器> [-i in][-o out][-p 变调][-g 增益dB][-c limit|soft|hard]`、
  `status`（worker/内存）、`ls [dir]`、`sdinfo`。例：`infer bass -p -12 -g 9 -c soft`。
