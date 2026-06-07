"""分帧与重叠相加（与 ESP32 流式帧布局一致：非中心、帧 = [i*hop, i*hop+frame)）。"""

from __future__ import annotations

import numpy as np


def num_frames(n_samples: int, frame_size: int, hop_size: int, pad_end: bool = True) -> int:
    """计算帧数。pad_end=True 时末尾补零，保证覆盖整段信号。"""
    if n_samples <= 0:
        return 0
    if pad_end:
        return int(np.ceil(n_samples / hop_size))
    if n_samples < frame_size:
        return 0
    return 1 + (n_samples - frame_size) // hop_size


def frame_signal(
    signal: np.ndarray,
    frame_size: int,
    hop_size: int,
    pad_end: bool = True,
) -> np.ndarray:
    """切帧，返回 (n_frames, frame_size)。

    非中心对齐：第 i 帧 = signal[i*hop : i*hop+frame_size]，与流式推理一致。
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = len(signal)
    nf = num_frames(n, frame_size, hop_size, pad_end=pad_end)
    if nf == 0:
        return np.zeros((0, frame_size), dtype=np.float32)

    total_needed = (nf - 1) * hop_size + frame_size
    if total_needed > n:
        signal = np.pad(signal, (0, total_needed - n), mode="constant")

    # 用 stride 构造重叠视图后再拷贝，避免共享内存副作用
    idx = np.arange(frame_size)[None, :] + hop_size * np.arange(nf)[:, None]
    frames = signal[idx]
    return np.ascontiguousarray(frames, dtype=np.float32)


def overlap_add(
    frames: np.ndarray,
    hop_size: int,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """重叠相加重建信号（重叠帧加权平均）。frames: (n_frames, frame_size)。

    传入的是“原始帧”（未加窗）；本函数对其加一次窗后按窗和归一化：
        out[n] = Σ_i frame_i[k]·win[k] / Σ_i win[k]
    这样当各帧来自同一连续信号时可精确重建，接缝处无增益起伏。
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError("frames 必须是 (n_frames, frame_size)")
    nf, frame_size = frames.shape
    out_len = (nf - 1) * hop_size + frame_size
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    if window is None:
        win = np.ones(frame_size, dtype=np.float32)
    else:
        win = np.asarray(window, dtype=np.float32)

    for i in range(nf):
        start = i * hop_size
        out[start : start + frame_size] += frames[i] * win
        norm[start : start + frame_size] += win

    nonzero = norm > 1e-8
    out[nonzero] /= norm[nonzero]
    return out


def hann_window(frame_size: int) -> np.ndarray:
    """周期性 Hann 窗（与 STFT 约定一致）。"""
    n = np.arange(frame_size)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame_size)).astype(np.float32)


__all__ = ["num_frames", "frame_signal", "overlap_add", "hann_window"]
