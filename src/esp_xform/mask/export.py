"""谱掩码模型导出：BNNW 网络权重 + 展开矩阵/统计/嵌入（.npy 与 C .h）。

权重序列化顺序严格对应 C 计算图 (MCU/Tinynn/model bnn_masknet) 的遍历顺序：
  c1,f1,c2,f2,c3,head_mag,head_phase,head_noise，每层 W 再 b；FiLM 的 W = fc.weight.T。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch.nn as nn

from ..train.export import (
    BNN_WMAGIC,
    BNN_WVER,
    write_bnn_weights_bin,
    write_bytes_as_c_array,
    write_c_array_1d,
    write_c_array_2d,
)
from .audio import build_all_matrices
from .model import FiLM

# ---- 统一模型产物容器 xform_model.bin ----
PKG_MAGIC = 0x4D524658      # 'XFRM' (小端)
PKG_VERSION = 1
PKG_DT_RAW, PKG_DT_F32, PKG_DT_I32 = 0, 1, 2
_PKG_TOC_ENTRY = 48          # name[16] + dtype(u32) + ndim(u32) + shape[4](u32) + offset(u32) + nbytes(u32)


def _collect_tensors(model) -> List[np.ndarray]:
    tensors: List[np.ndarray] = []

    def w(t):
        return t.detach().cpu().numpy().astype(np.float32)

    for mod in model.ordered_modules():
        if isinstance(mod, nn.Conv1d):
            tensors.append(w(mod.weight).reshape(mod.out_channels, -1).reshape(-1))
            tensors.append(w(mod.bias).reshape(-1))
        elif isinstance(mod, FiLM):
            tensors.append(w(mod.fc.weight).T.copy().reshape(-1))  # [2C,cond] -> [cond,2C]
            tensors.append(w(mod.fc.bias).reshape(-1))
        else:
            raise TypeError(f"未知可导出模块: {type(mod)}")
    return tensors


def export_masknet_weights(model, out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate([t.reshape(-1) for t in _collect_tensors(model)]).astype("<f4")
    bin_path = out / "masknet_weights.bin"
    write_bnn_weights_bin(bin_path, flat)
    h_path = out / "masknet_weights.h"
    write_bytes_as_c_array(h_path, "masknet_weights_bin", bin_path.read_bytes())
    return {"weights_bin": str(bin_path), "weights_h": str(h_path), "num_params": str(int(flat.size))}


def _bnnw_bytes(model) -> bytes:
    """BNNW 权重字节流 (magic+ver+count+float32[]), 可直接喂 bnn_*_load_weights_mem。"""
    flat = np.concatenate([t.reshape(-1) for t in _collect_tensors(model)]).astype("<f4")
    return struct.pack("<II", BNN_WMAGIC, BNN_WVER) + struct.pack("<Q", int(flat.size)) + flat.tobytes()


# ---- 图 IR 段 (数据驱动建图, 与 C bnn_masknet build_graph 等价) ----
# 复用共享 IR 写入器 (esp_xform.onnx.ir_writer), 与 ONNX 转换器同一套格式, 避免重复维护。
from ..onnx import ir_writer as _iw           # noqa: E402

IR_MAGIC = _iw.IR_MAGIC                        # 兼容旧引用
PKG_BLOCK_FRAMES = 64                          # 必须与固件 BLOCK_FRAMES 一致
_DPHI_MAX = 1.57079632679                      # C 默认 dphi_max=π/2 (固件两条建图路径都用此值)


def _masknet_graph_ir(cfg, emb_dim: int, block_frames: int = PKG_BLOCK_FRAMES) -> bytes:
    """构建与 C build_graph 等价的图 IR (节点表 + 输出头, 不含权重)。
    权重仍由 'weights' 段提供 —— 图拓扑/层内顺序与 _collect_tensors 完全一致。
    输出头约定: [0]=幅度掩码 [1]=相位残差 [2]=噪声带。"""
    s, m = cfg.stft, cfg.model
    Mn, H, k, T, E = s.n_mels, m.hidden, m.kernel, block_frames, emb_dim
    pad1 = (k - 1) // 2
    pad2 = m.dilation2 * (k - 1) // 2
    nd = _iw.ir_node

    nodes = [
        _iw.input_node([1, Mn, T]),                                                              # 0 mel
        _iw.input_node([1, E]),                                                                  # 1 cond/emb
        nd("conv1d", 1, inc=Mn, outc=H, k=k, st=1, pad=pad1, dil=1, ndep=1, a=(0,)),             # 2 c1
        nd("activation", 1, act=_iw.ACT_RELU, ndep=1, a=(2,)),                                   # 3
        nd("film", 1, outc=H, inf=E, ndep=2, a=(3, 1), extra0=0),                                # 4 f1
        nd("conv1d", 1, inc=H, outc=H, k=k, st=1, pad=pad2, dil=m.dilation2, ndep=1, a=(4,)),    # 5 c2
        nd("activation", 1, act=_iw.ACT_RELU, ndep=1, a=(5,)),                                   # 6
        nd("film", 1, outc=H, inf=E, ndep=2, a=(6, 1), extra0=0),                                # 7 f2
        nd("conv1d", 1, inc=H, outc=Mn, k=k, st=1, pad=pad1, dil=1, ndep=1, a=(7,)),             # 8 c3
        nd("activation", 1, act=_iw.ACT_RELU, ndep=1, a=(8,)),                                   # 9 a3
        nd("conv1d", 1, inc=Mn, outc=Mn, k=1, st=1, pad=0, dil=1, ndep=1, a=(9,)),               # 10 head_mag conv
        nd("activation", 1, act=_iw.ACT_SIGMOID, param=m.gmax, ndep=1, a=(10,)),                 # 11 mask
        nd("conv1d", 1, inc=Mn, outc=m.phase_bands, k=1, st=1, pad=0, dil=1, ndep=1, a=(9,)),    # 12 head_phase conv
        nd("activation", 1, act=_iw.ACT_TANH, param=_DPHI_MAX, ndep=1, a=(12,)),                 # 13 phase
        nd("conv1d", 1, inc=Mn, outc=m.noise_bands, k=1, st=1, pad=0, dil=1, ndep=1, a=(9,)),    # 14 head_noise conv
        nd("activation", 1, act=_iw.ACT_SOFTPLUS, param=1.0, ndep=1, a=(14,)),                   # 15 noise
    ]
    return _iw.pack_ir(nodes, [11, 13, 15])    # 权重在 'weights' 段, IR 内不嵌


def export_model_package(model, cfg, mel_mean, mel_std,
                         instruments: Dict[str, int], out_path: str | Path) -> Dict[str, str]:
    """统一模型产物容器 xform_model.bin (单文件, 自带 TOC), 供 MCU model_store 解析。

    段: config(i32) / weights(BNNW raw) / mel_mean(f32) / mel_std(f32) /
        emb(f32 [N,E]) / names(乐器名, \\n 分隔)。新增乐器只需重导出, 固件不改。
    """
    s, m = cfg.stft, cfg.model
    emb = model.emb.weight.detach().cpu().numpy().astype("<f4")
    num_inst, emb_dim = int(emb.shape[0]), int(emb.shape[1])
    id2name = {int(v): k for k, v in instruments.items()}
    names = ("\n".join(id2name.get(i, f"inst{i}") for i in range(num_inst)) + "\n").encode("utf-8")
    config = np.array([PKG_VERSION, s.sample_rate, s.n_fft, s.hop, s.n_mels,
                       int(s.fmin), int(s.fmax), int(round(m.gmax * 1000)),
                       m.phase_bands, m.noise_bands, m.hidden, m.kernel, m.dilation2,
                       emb_dim, num_inst, 0], dtype="<i4")
    mel_mean = np.asarray(mel_mean, dtype="<f4").reshape(-1)
    mel_std = np.asarray(mel_std, dtype="<f4").reshape(-1)

    graph_ir = _masknet_graph_ir(cfg, emb_dim)

    # (name, dtype, shape, data_bytes)
    sections = [
        ("config",   PKG_DT_I32, [int(config.size)],     config.tobytes()),
        ("weights",  PKG_DT_RAW, [0],                     _bnnw_bytes(model)),
        ("mel_mean", PKG_DT_F32, [int(mel_mean.size)],    mel_mean.tobytes()),
        ("mel_std",  PKG_DT_F32, [int(mel_std.size)],     mel_std.tobytes()),
        ("emb",      PKG_DT_F32, [num_inst, emb_dim],     emb.tobytes()),
        ("names",    PKG_DT_RAW, [len(names)],            names),
        ("graph",    PKG_DT_RAW, [len(graph_ir)],         graph_ir),   # 数据驱动建图 IR
    ]
    n = len(sections)
    cur = 16 + n * _PKG_TOC_ENTRY
    toc = b""
    blobs = []
    for name, dt, shape, data in sections:
        cur += (-cur) % 4                                   # 4 字节对齐
        sh = (list(shape) + [0, 0, 0, 0])[:4]
        nm = name.encode("ascii")[:16].ljust(16, b"\x00")
        toc += nm + struct.pack("<II", dt, len(shape)) + struct.pack("<4I", *sh) \
               + struct.pack("<II", cur, len(data))
        blobs.append((cur, data))
        cur += len(data)

    buf = bytearray(struct.pack("<IIII", PKG_MAGIC, PKG_VERSION, n, 0))
    buf += toc
    for off, data in blobs:
        if len(buf) < off:
            buf += b"\x00" * (off - len(buf))
        buf += data
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf)
    return {"model_package": str(out), "bytes": str(len(buf)), "num_instruments": str(num_inst)}


def export_masknet_artifacts(model, cfg, out_dir: str | Path,
                             instruments: Dict[str, int], proc_dir: str | Path = None) -> Dict[str, str]:
    """导出权重 + 展开矩阵 + 统计 + 乐器嵌入 + 配置，供 C 端 bnn_masknet 加载。"""
    out = Path(out_dir)
    exp = out / "exports"
    exp.mkdir(parents=True, exist_ok=True)

    result: Dict[str, str] = {}
    result.update(export_masknet_weights(model, exp))

    # 展开矩阵
    mats = build_all_matrices(cfg)
    write_c_array_2d(exp / "mel_inv.h", "mel_inv", mats["mel_inv"])          # [n_bins,M]
    write_c_array_2d(exp / "phase_inv.h", "phase_inv", mats["phase_inv"])    # [n_bins,P]
    write_c_array_2d(exp / "noise_fb.h", "noise_fb", mats["noise_fb"])       # [B,n_bins]
    write_c_array_2d(exp / "mel_basis.h", "mel_basis", mats["mel_basis"])    # [M,n_bins]
    for k, v in mats.items():
        np.save(exp / f"{k}.npy", v)

    # 梅尔标准化统计 (优先从 proc_dir 读，回退占位)
    if proc_dir and (Path(proc_dir) / "stats.npz").exists():
        st = np.load(Path(proc_dir) / "stats.npz")
        mel_mean, mel_std = st["mel_mean"], st["mel_std"]
    else:
        mel_mean = np.zeros(cfg.stft.n_mels, np.float32)
        mel_std = np.ones(cfg.stft.n_mels, np.float32)
    write_c_array_1d(exp / "mel_mean.h", "mel_mean", mel_mean)
    write_c_array_1d(exp / "mel_std.h", "mel_std", mel_std)
    np.save(exp / "mel_mean.npy", mel_mean)
    np.save(exp / "mel_std.npy", mel_std)

    # 乐器嵌入表
    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    write_c_array_2d(exp / "instrument_embeddings.h", "instrument_embeddings", emb)
    np.save(exp / "instrument_embeddings.npy", emb)

    # 统一模型产物容器 (MCU 从 TF 卡读取这一个文件即可)
    pkg = export_model_package(model, cfg, mel_mean, mel_std, instruments, exp / "xform_model.bin")
    result["model_package"] = pkg["model_package"]

    # 配置快照
    (exp / "mask_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (exp / "instruments.json").write_text(
        json.dumps(instruments, indent=2, ensure_ascii=False), encoding="utf-8")

    result.update({
        "mel_inv_h": str(exp / "mel_inv.h"),
        "phase_inv_h": str(exp / "phase_inv.h"),
        "noise_fb_h": str(exp / "noise_fb.h"),
        "instrument_embeddings_h": str(exp / "instrument_embeddings.h"),
        "config_json": str(exp / "mask_config.json"),
    })
    return result


__all__ = ["export_masknet_weights", "export_masknet_artifacts", "export_model_package"]
