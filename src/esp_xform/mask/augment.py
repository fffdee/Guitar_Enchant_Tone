"""域增强（对应 §4.4）：仅作用于"输入吉他"，缩小渲染↔真实拾音的域差。

噪声 / 随机峰值 EQ / 增益 / 轻微 detune。目标标签不变；因数据集对每个窗即时重算
掩码标签(Ymel/(Xmel+eps))，对输入增强后标签自动自洽（目标不变、输入变化）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AugmentConfig:
    enable: bool = True
    p_noise: float = 0.7
    snr_db: Tuple[float, float] = (30.0, 50.0)
    p_gain: float = 0.7
    gain_db: Tuple[float, float] = (-6.0, 6.0)
    p_eq: float = 0.5
    eq_freq: Tuple[float, float] = (200.0, 6000.0)
    eq_gain_db: Tuple[float, float] = (-6.0, 6.0)
    eq_q: float = 1.0
    p_detune: float = 0.2
    detune_cents: Tuple[float, float] = (-15.0, 15.0)


def _biquad_peaking(x: np.ndarray, sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ 峰值 EQ 双二阶滤波，直接形式 I。"""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * max(q, 1e-3))
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        xi = x[i]
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        y[i] = yi
    return y


def _detune(x: np.ndarray, cents: float) -> np.ndarray:
    """轻微变调（线性重采样后裁/补到原长，近似微音高漂移）。"""
    ratio = 2.0 ** (cents / 1200.0)
    n = len(x)
    src = np.arange(n) / ratio
    src = np.clip(src, 0, n - 1)
    y = np.interp(src, np.arange(n), x)
    return y.astype(x.dtype)


def augment_input(x: np.ndarray, sr: int, cfg: AugmentConfig,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """对输入吉他波形施加随机增强，返回同长度波形。"""
    if not cfg.enable:
        return x
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float64).copy()

    if rng.random() < cfg.p_eq:
        f0 = float(rng.uniform(*cfg.eq_freq))
        g = float(rng.uniform(*cfg.eq_gain_db))
        y = _biquad_peaking(y, sr, f0, g, cfg.eq_q)
    if rng.random() < cfg.p_detune:
        y = _detune(y, float(rng.uniform(*cfg.detune_cents)))
    if rng.random() < cfg.p_gain:
        y = y * (10.0 ** (float(rng.uniform(*cfg.gain_db)) / 20.0))
    if rng.random() < cfg.p_noise:
        snr = float(rng.uniform(*cfg.snr_db))
        sig_p = float(np.mean(y * y) + 1e-12)
        noise_p = sig_p / (10.0 ** (snr / 10.0))
        y = y + rng.standard_normal(len(y)) * np.sqrt(noise_p)
    return y.astype(np.float32)


__all__ = ["AugmentConfig", "augment_input"]
