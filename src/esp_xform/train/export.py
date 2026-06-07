"""导出训练产物：checkpoint、乐器嵌入、特征归一化参数（.npy 与 C 头文件）。

为后续 ESP32 固件准备：嵌入矩阵与特征 mean/std 直接生成可 #include 的 C 数组。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# BNNW 权重二进制魔数 (须与 C 端 bnn_graph.c 的 BNN_WMAGIC 数值一致)
BNN_WMAGIC = 0x57574E42
BNN_WVER = 1


def _fmt_floats(values, per_line: int = 8) -> str:
    items = [f"{v:.8f}f" for v in values]
    lines = [
        "    " + ", ".join(items[i : i + per_line])
        for i in range(0, len(items), per_line)
    ]
    return ",\n".join(lines)


def write_c_array_1d(path: Path, var_name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    body = _fmt_floats(arr.tolist())
    text = (
        f"// 自动生成，请勿手改\n"
        f"#define {var_name.upper()}_LEN {arr.size}\n"
        f"static const float {var_name}[{arr.size}] = {{\n{body}\n}};\n"
    )
    path.write_text(text, encoding="utf-8")


def write_c_array_2d(path: Path, var_name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    rows, cols = arr.shape
    row_strs = ["    {" + ", ".join(f"{v:.8f}f" for v in row) + "}" for row in arr]
    body = ",\n".join(row_strs)
    text = (
        f"// 自动生成，请勿手改\n"
        f"#define {var_name.upper()}_ROWS {rows}\n"
        f"#define {var_name.upper()}_COLS {cols}\n"
        f"static const float {var_name}[{rows}][{cols}] = {{\n{body}\n}};\n"
    )
    path.write_text(text, encoding="utf-8")


def _collect_bnn_tensors(model) -> List[np.ndarray]:
    """按 C 计算图(bnn_xform build_graph)遍历顺序收集权重张量(已展平为 1D float32)。

    顺序：对每个卷积块 i —— conv[i].W, conv[i].b，若该块带 FiLM 则紧随 film.W, film.b；
    最后 head.W, head.b。务必与 MCU/Tinynn/model/src/bnn_xform.c 的构图顺序一致。

    权重布局对齐 C 端：
      - conv1d.W: PyTorch [Cout,Cin,K] -> reshape [Cout, Cin*K] (C-order, 内层 ci*K+k)
      - film.W : PyTorch nn.Linear 权重 [2C,E] -> 转置为 [E,2C] (C 端 gb=e·W[E,2C])
      - head.W : [out,prev,1] -> [out, prev]
    """
    tensors: List[np.ndarray] = []

    def w(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    convs = model.convs
    films = model.films
    has_film = model.has_film
    for i, conv in enumerate(convs):
        tensors.append(w(conv.weight).reshape(conv.out_channels, -1).reshape(-1))
        tensors.append(w(conv.bias).reshape(-1))
        if has_film[i]:
            film = films[i]
            proj_w = w(film.proj.weight)              # [2C, E]
            tensors.append(proj_w.T.copy().reshape(-1))  # -> [E, 2C] flat
            tensors.append(w(film.proj.bias).reshape(-1))

    head = model.head
    tensors.append(w(head.weight).reshape(head.out_channels, -1).reshape(-1))
    tensors.append(w(head.bias).reshape(-1))
    return tensors


def write_bnn_weights_bin(path: Path, flat: np.ndarray) -> int:
    """写出 BNNW 二进制：magic(u32) | ver(u32) | num_params(u64) | float32[num_params]。

    与 C 端 bnn_graph_load_weights / bnn_graph_load_weights_mem 完全兼容。返回字节数。
    """
    flat = np.ascontiguousarray(flat, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", BNN_WMAGIC))
        f.write(struct.pack("<I", BNN_WVER))
        f.write(struct.pack("<Q", int(flat.size)))
        f.write(flat.tobytes())
    return 16 + flat.size * 4


def write_bytes_as_c_array(path: Path, var_name: str, data: bytes, per_line: int = 16) -> None:
    """把任意字节写成可 #include 的 C 数组 (无文件系统的 MCU 直接用)。"""
    lines = []
    for i in range(0, len(data), per_line):
        chunk = data[i : i + per_line]
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk))
    body = ",\n".join(lines)
    text = (
        f"// 自动生成，请勿手改\n"
        f"#define {var_name.upper()}_LEN {len(data)}\n"
        f"static const unsigned char {var_name}[{len(data)}] = {{\n{body}\n}};\n"
    )
    path.write_text(text, encoding="utf-8")


def export_bnn_weights(model, output_dir: str | Path) -> Dict[str, str]:
    """导出 ConditionalDDSPNet 的网络权重为 BNNW (.bin) 与 C 数组 (.h)。

    供 C 端 bnn_xform_load_weights_mem 加载。返回 {bin, h, num_params}。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = _collect_bnn_tensors(model)
    flat = np.concatenate([t.reshape(-1) for t in tensors]).astype("<f4")

    bin_path = out / "xform_weights.bin"
    write_bnn_weights_bin(bin_path, flat)

    h_path = out / "xform_weights.h"
    write_bytes_as_c_array(h_path, "xform_weights_bin", bin_path.read_bytes())

    return {
        "weights_bin": str(bin_path),
        "weights_h": str(h_path),
        "num_params": str(int(flat.size)),
    }


def export_artifacts(
    model,
    cfg,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    instruments: Dict,
    output_dir: str | Path,
    extra: Optional[Dict] = None,
) -> Dict[str, str]:
    """导出所有产物，返回各文件路径。"""
    out = Path(output_dir)
    exp = out / "exports"
    ckpt_dir = out / "checkpoints"
    exp.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) checkpoint
    ckpt_path = ckpt_dir / "model_final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "num_instruments": model.num_instruments,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "instruments": instruments,
            "extra": extra or {},
        },
        ckpt_path,
    )

    # 2) 乐器嵌入
    emb = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    np.save(exp / "instrument_embeddings.npy", emb)
    write_c_array_2d(exp / "instrument_embeddings.h", "instrument_embeddings", emb)

    # 3) 特征归一化参数
    np.save(exp / "feature_mean.npy", feature_mean)
    np.save(exp / "feature_std.npy", feature_std)
    write_c_array_1d(exp / "feature_mean.h", "feature_mean", feature_mean)
    write_c_array_1d(exp / "feature_std.h", "feature_std", feature_std)

    # 4) 配置与乐器映射快照
    (exp / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (exp / "instruments.json").write_text(
        json.dumps(instruments, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    result = {
        "checkpoint": str(ckpt_path),
        "embeddings_npy": str(exp / "instrument_embeddings.npy"),
        "embeddings_h": str(exp / "instrument_embeddings.h"),
        "feature_mean_h": str(exp / "feature_mean.h"),
        "feature_std_h": str(exp / "feature_std.h"),
        "config_json": str(exp / "config.json"),
    }

    # 5) MCU 部署用网络权重 (BNNW .bin + .h)，供 bnn_xform_load_weights_mem 加载
    if hasattr(model, "convs") and hasattr(model, "head"):
        try:
            result.update(export_bnn_weights(model, exp))
        except Exception as e:  # 导出权重失败不应阻断其余产物
            print(f"[export] BNNW 权重导出跳过: {e}")

    return result


__all__ = [
    "export_artifacts",
    "export_bnn_weights",
    "write_bnn_weights_bin",
    "write_bytes_as_c_array",
    "write_c_array_1d",
    "write_c_array_2d",
]
