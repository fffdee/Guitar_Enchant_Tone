"""失真效果器（传统 DSP，对应 §7）。波形整形 + 可选过采样抗混叠。

与神经网络音色转换正交：零 NN 算力，可串/并联。C 端镜像见 MCU/Tinynn/effect/bnn_fx。
shaper: tanh 软削波 / cubic 软削波 / hard 硬削波。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _fir_lowpass(taps: int, fc: float) -> np.ndarray:
    """窗口 sinc 低通，归一化(和=1)。fc 为归一化截止(cycles/sample)。"""
    n = np.arange(taps)
    c = (taps - 1) / 2.0
    h = np.sinc(2.0 * fc * (n - c)) * np.hamming(taps)
    return (h / h.sum()).astype(np.float64)


def waveshape(x: np.ndarray, kind: str = "tanh", drive: float = 2.0) -> np.ndarray:
    u = drive * np.asarray(x, dtype=np.float64)
    if kind == "tanh":
        return np.tanh(u)
    if kind == "cubic":
        uc = np.clip(u, -1.0, 1.0)
        return (uc - uc ** 3 / 3.0) * 1.5
    if kind == "hard":
        return np.clip(u, -1.0, 1.0)
    return u


def distort(x: np.ndarray, kind: str = "tanh", drive: float = 2.0,
            out_level: float = 1.0, mix: float = 1.0, oversample: int = 1) -> np.ndarray:
    """整段失真。oversample∈{1,2,4}：过采样→整形→抗混叠→抽取。"""
    x = np.asarray(x, dtype=np.float64)
    if oversample <= 1:
        sh = waveshape(x, kind, drive)
    else:
        L = int(oversample)
        taps = 16 * L + 1
        h = _fir_lowpass(taps, 0.5 / L)
        up = np.zeros(len(x) * L, dtype=np.float64)
        up[::L] = x
        up = np.convolve(up, h * L, mode="same")     # 插值低通(增益 L)
        sh_os = waveshape(up, kind, drive)
        sh_os = np.convolve(sh_os, h, mode="same")   # 抗混叠低通
        sh = sh_os[::L]
    y = (1.0 - mix) * x + mix * sh
    return (y * out_level).astype(np.float32)


@dataclass
class DistortionConfig:
    kind: str = "tanh"
    drive: float = 2.0
    out_level: float = 1.0
    mix: float = 1.0
    oversample: int = 2


class Distortion:
    def __init__(self, cfg: Optional[DistortionConfig] = None) -> None:
        self.cfg = cfg or DistortionConfig()

    def process(self, x: np.ndarray) -> np.ndarray:
        c = self.cfg
        return distort(x, c.kind, c.drive, c.out_level, c.mix, c.oversample)


__all__ = ["waveshape", "distort", "DistortionConfig", "Distortion"]
