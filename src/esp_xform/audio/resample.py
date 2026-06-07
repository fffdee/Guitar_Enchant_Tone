"""采样率转换（纯 numpy，傅里叶法）。

真实录音/渲染常见 44.1k/96k 等，需统一到项目的 48 kHz。这里用频域法
（等价 scipy.signal.resample）：rfft → 截断/补零谱 → irfft，O(N log N)。
对整段离线重采样质量足够；若需更高质量可后续替换为多相 FIR。
"""

from __future__ import annotations

import numpy as np


def resample_fft(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """把 x 从 orig_sr 重采样到 target_sr。x 为单声道 float。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if orig_sr == target_sr or x.size == 0:
        return x.astype(np.float32)

    n = x.size
    new_n = int(round(n * target_sr / orig_sr))
    if new_n <= 0:
        return np.zeros(0, dtype=np.float32)

    X = np.fft.rfft(x)
    Y = np.zeros(new_n // 2 + 1, dtype=complex)
    k = min(X.size, Y.size)
    Y[:k] = X[:k]
    # 若为下采样，截断高频即天然抗混叠；Nyquist 分量减半更稳妥
    if new_n < n and k > 0:
        Y[k - 1] *= 0.5
    y = np.fft.irfft(Y, n=new_n) * (new_n / n)
    return y.astype(np.float32)


def resample_to(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """语义化封装：返回 target_sr 下的信号。"""
    return resample_fft(x, orig_sr, target_sr)


__all__ = ["resample_fft", "resample_to"]
