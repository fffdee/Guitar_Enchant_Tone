"""源↔目标对齐：用互相关估计样点级延迟并补偿。

真实渲染/录音中，源吉他与目标乐器可能存在固定延迟差（DAW 插件延迟等），
会破坏帧级对齐。这里用 FFT 互相关估计整数延迟并移位补偿。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def estimate_lag(reference: np.ndarray, signal: np.ndarray, max_lag: int | None = None) -> int:
    """估计 signal 相对 reference 的延迟（样点）。

    返回 lag：将 signal 向左移 lag（即 shift -lag）可与 reference 对齐。
    lag>0 表示 signal 落后于 reference（开头多了 lag 样点）。
    """
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(signal, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0

    nfft = _next_pow2(na + nb - 1)
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    # 标准互相关 c[k]=Σ a[n]·b[n+k]，DFT 为 conj(A)·B；
    # 峰值 k>0 表示 b（signal）相对 a（reference）延迟 k 个样点
    corr = np.fft.irfft(np.conj(A) * B, nfft)

    # corr[k] 对应 lag=k（0..nfft-1），其中 k 大于 nfft/2 视为负延迟
    lags = np.arange(nfft)
    neg = lags > nfft // 2
    lag_signed = lags.copy()
    lag_signed[neg] = lags[neg] - nfft

    if max_lag is not None:
        valid = np.abs(lag_signed) <= max_lag
        corr = np.where(valid, corr, -np.inf)

    best = int(np.argmax(corr))
    return int(lag_signed[best])


def shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    """非循环移位：shift>0 右移（前补零），shift<0 左移（后补零）。"""
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    if shift == 0:
        return x.copy()
    if shift > 0:
        out[shift:] = x[: len(x) - shift]
    else:
        out[: len(x) + shift] = x[-shift:]
    return out


def align_to_reference(
    reference: np.ndarray, signal: np.ndarray, max_lag: int | None = None
) -> Tuple[np.ndarray, int]:
    """把 signal 对齐到 reference（同长），返回 (对齐后信号, 估计 lag)。"""
    lag = estimate_lag(reference, signal, max_lag)
    aligned = shift_signal(signal, -lag)
    # 统一长度到 reference
    n = len(reference)
    if len(aligned) < n:
        aligned = np.pad(aligned, (0, n - len(aligned)))
    else:
        aligned = aligned[:n]
    return aligned.astype(np.float32), lag


__all__ = ["estimate_lag", "shift_signal", "align_to_reference"]
