# Guitar Enchant Tone

**ESP32-P4 实时吉他音色转换系统**

---

## 项目简介

Guitar Enchant Tone 是一套运行在 **ESP32-P4** 上的实时吉他音色转换系统。插入吉他，即可实时转换为 Bass、钢琴、提琴等乐器音色，切换乐器只需一条命令，无需重载模型。

核心技术路线：**频域谱掩码（Spectral Mask）**——不做音高估计，直接用 CNN 预测梅尔域增益掩码作用于输入频谱，天然支持和弦/复音，且算力固定、与发声音符数无关。

---

## 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 模型训练（PC） | ✅ | PyTorch MaskNet + FiLM，3头输出 |
| 模型导出 | ✅ | `.bin` 格式（F32 + INT8权重 + 矩阵/嵌入） |
| MCU 推理框架（tinynn） | ✅ | 自研 C 推理框架，含 IR 建图、INT8 加速 |
| 音频前端（specfront） | ✅ | RFFT + 稀疏梅尔滤波 |
| 音频合成（specsynth） | ✅ | 稀疏掩码展开 + 相位旋转 + OLA |
| ESP-DSP 加速 | ✅ | PIE 硬件循环 FFT/向量运算 |
| ESP-NN INT8 加速 | ✅ | Conv1d INT8路径（dilation=1层） |
| CLI 控制 | ✅ | 串口命令行，支持变调/增益/噪声开关 |
| SD 卡读写 | ✅ | WAV 读入 → 推理 → WAV 写出 |
| 实时推理性能 | ✅ | **1.74× RT**（11.3s 推理 20s 音频，稀疏优化后）|

---

## 快速上手

### 硬件
- ESP32-P4 开发板
- SD 卡（挂载于 `/sdcard`）
- 串口终端（115200）

### 上手步骤

```bash
# 1. 训练并导出模型
cd src/esp_xform/mask
conda activate tcn_cuda
python export.py --model checkpoints/best.pt --out /sdcard/model/

# 或使用一键导出脚本（Windows）
export_model.bat

# 2. 将 xform_model.bin 放到 SD 卡 /sdcard/model/ 目录
# 3. 将待转换的 guitar.wav (48kHz单声道) 放到 /sdcard/in/

# 4. 烧录固件，连接串口，输入推理命令
infer bass -p -12 -g 9 -c soft
```

### CLI 常用命令

```
infer <乐器> [-p 变调半音] [-g 增益dB] [-c limit|soft|hard] [-N 开噪声注入]
status        # 查看推理进度与内存
stats         # 查看推理时间统计
model         # 打印已加载模型信息
```

---

## 技术概述

```
吉他 WAV
  │
  ▼  (每帧 1024点, 帧移 256点)
┌─────────────────────────────────────────────────────────────────┐
│ 频域前端 (specfront)                                             │
│   Hann 加窗 → RFFT → 幅度谱[513] + 单位相位{cos,sin}[513×2]      │
│   稀疏梅尔滤波 → 对数梅尔[96]                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ logmel[96,T=64]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 神经网络推理 (MaskNet, tinynn)                                    │
│   CNN主干: Conv1D 96→128→128→96 + FiLM(乐器嵌入16维)             │
│   输出 ①掩码[96,T]  ②相位残差[64,T]  ③噪声带[16,T]              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ per-frame: mask[96] dphi[64] noise[16]
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ 频域合成 (specsynth)                                              │
│   稀疏 mel_inv → 线性增益[513]                                    │
│   稀疏 phase_inv → 相位残差展开[513]                              │
│   复数旋转: Y = gain × X × exp(j·(φ + Δφ))                      │
│   (可选) 噪声注入                                                 │
│   IRFFT + 加窗 OLA → 时域 hop[256]                               │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
目标乐器 WAV
```

详细文档见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

---

## 工程结构

```
Guitar_Enchant_Tone/
├── src/esp_xform/mask/     # PC端: 训练、推理验证、导出
│   ├── model.py            # MaskNet + FiLM
│   ├── train.py            # 训练循环
│   ├── audio.py            # STFT/梅尔/展开矩阵工具
│   └── export.py           # 模型→.bin 导出
├── scripts/
│   └── export_bin.py       # 命令行导出入口
├── export_model.bat        # Windows一键导出脚本
├── Reaper-MCP/             # Reaper DAW 自动化控制 (数据采集)
│   ├── reaper_mcp/tools/   # MCP 工具集 (轨道/MIDI/FX/渲染)
│   └── docs/               # 工具使用文档
├── MCU/tiny_nn/            # ESP32-P4 固件
│   ├── components/tinynn/  # 自研推理框架
│   │   ├── frontend/       # bnn_specfront (RFFT + 梅尔)
│   │   ├── synth/          # bnn_specsynth (合成 + OLA)
│   │   ├── model/          # bnn_masknet (推理编排)
│   │   ├── layer/          # Conv1d (F32 + INT8)
│   │   ├── graph/          # 计算图执行引擎
│   │   └── operator/       # 算子后端 (CPU/DSP/NN)
│   └── components/bsp/     # 板级支持 + 应用层
│       └── 06_app/         # audio_xform, infer_worker, cli
├── IMPLEMENTATION.md       # 详细设计文档
└── README.md               # 本文件
```

---

## 训练数据采集

使用 **Reaper DAW + Ample Sound** 采集训练数据，通过 `Reaper-MCP/` 下的 MCP 工具可让 Cursor AI 自动化控制 Reaper（批量写 MIDI、渲染 Stems、管理工程）。

核心采集步骤：
1. Reaper 工程设置为 48kHz，各轨挂载对应 Ample Sound VSTi（Guitar / Bass / Ukulele 等）。
2. 所有乐器轨共享同一份 MIDI（复制 MIDI item 或 MIDI 路由）。
3. 关闭 Ample Sound Humanize、关闭 Strummer 时间随机。
4. `File → Render → Stems (selected tracks)` 一次性导出全部对齐 WAV。

详细操作见 [IMPLEMENTATION.md](IMPLEMENTATION.md) §1 及附录 C。
