"""YIN 基频估计（纯 numpy，便于 1:1 移植到 ESP32 C）。

实现 de Cheveigné & Kawahara (2002) YIN 的核心步骤：
  1) 差分函数 d(tau)
  2) 累积均值归一化差分 CMND d'(tau)
  3) 绝对阈值选 tau
  4) 抛物线插值精修
差分函数中的互相关项用 np.correlate 计算（C 端可用自相关/FFT 等价实现）。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-12


def _cmnd(frame: np.ndarray, w: int, tau_max: int) -> np.ndarray:
    """计算累积均值归一化差分函数 d'(tau)，tau ∈ [0, tau_max]。"""
    x = frame.astype(np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(x * x)])
    term1 = cumsum[w] - cumsum[0]                      # Σ_{j<w} x[j]^2 （常数）
    taus = np.arange(tau_max + 1)
    term2 = cumsum[taus + w] - cumsum[taus]            # Σ_{j<w} x[j+tau]^2
    cross = np.correlate(x[: w + tau_max], x[:w], mode="valid")[: tau_max + 1]
    d = term1 + term2 - 2.0 * cross                    # 差分函数 d(tau)
    d = np.maximum(d, 0.0)

    d_prime = np.ones_like(d)
    running = np.cumsum(d[1:])
    denom = np.maximum(running, _EPS)
    d_prime[1:] = d[1:] * np.arange(1, len(d)) / denom
    return d_prime


def _parabolic_interp(d: np.ndarray, tau: int) -> float:
    """对 tau 处做抛物线插值，返回精修后的（亚采样）lag。"""
    if tau <= 0 or tau >= len(d) - 1:
        return float(tau)
    a, b, c = d[tau - 1], d[tau], d[tau + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < _EPS:
        return float(tau)
    return float(tau) + 0.5 * (a - c) / denom


def yin_f0(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = 55.0,
    f0_max: float = 1320.0,
    threshold: float = 0.1,
    integration_window: int | None = None,
    silence_rms: float = 1e-4,
) -> Tuple[float, float, float]:
    """对单帧估计基频。

    返回 (f0_hz, voiced_flag, aperiodicity)：
      - f0_hz：估计基频（无声/无周期时为 0）
      - voiced_flag：1=有声，0=无声/无周期
      - aperiodicity：所选 tau 处的 d' 值（越小越周期）
    """
    frame = np.asarray(frame, dtype=np.float64).reshape(-1)
    n = len(frame)

    rms = float(np.sqrt(np.mean(frame**2))) if n else 0.0
    if rms < silence_rms:
        return 0.0, 0.0, 1.0

    tau_min = max(2, int(np.floor(sample_rate / f0_max)))
    tau_max = int(np.ceil(sample_rate / f0_min))

    w = integration_window if integration_window is not None else n // 2
    if w + tau_max >= n:               # 保证 x[:w+tau_max] 不越界
        w = n - tau_max - 1
    if w <= 1:                          # 帧太短，无法估计
        return 0.0, 0.0, 1.0

    d_prime = _cmnd(frame, w, tau_max)

    search = d_prime.copy()
    search[:tau_min] = np.inf          # 屏蔽过小 lag（过高频）

    tau = -1
    for t in range(tau_min, tau_max + 1):
        if search[t] < threshold:
            # 收敛到该阈值谷的局部极小
            while t + 1 <= tau_max and search[t + 1] < search[t]:
                t += 1
            tau = t
            break
    if tau < 0:                         # 无低于阈值者，取全局最小
        tau = int(np.argmin(search[tau_min : tau_max + 1])) + tau_min

    aperiodicity = float(d_prime[tau])
    tau_ref = _parabolic_interp(d_prime, tau)
    if tau_ref <= 0:
        return 0.0, 0.0, aperiodicity

    f0 = sample_rate / tau_ref
    voiced = 1.0 if (aperiodicity < 0.3 and f0_min <= f0 <= f0_max) else 0.0
    if not (f0_min <= f0 <= f0_max):
        f0 = float(np.clip(f0, f0_min, f0_max))
    return float(f0), voiced, aperiodicity


def yin_f0_frames(
    frames: np.ndarray,
    sample_rate: int,
    f0_min: float = 55.0,
    f0_max: float = 1320.0,
    threshold: float = 0.1,
    integration_window: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一批帧 (n_frames, frame_size) 估计基频，逐帧返回数组。"""
    frames = np.asarray(frames, dtype=np.float64)
    nf = frames.shape[0]
    f0 = np.zeros(nf, dtype=np.float32)
    voiced = np.zeros(nf, dtype=np.float32)
    aper = np.ones(nf, dtype=np.float32)
    for i in range(nf):
        f0[i], voiced[i], aper[i] = yin_f0(
            frames[i], sample_rate, f0_min, f0_max, threshold, integration_window
        )
    return f0, voiced, aper


__all__ = ["yin_f0", "yin_f0_frames"]
