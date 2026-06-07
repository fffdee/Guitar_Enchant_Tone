"""幅度归一化与 dB 互转。"""

from __future__ import annotations

import numpy as np

EPS = 1e-7


def amp_to_db(amp: np.ndarray | float, eps: float = EPS) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.abs(amp), eps))


def db_to_amp(db: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (db / 20.0)


def peak_normalize(signal: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """峰值归一化到 ±peak。"""
    signal = np.asarray(signal, dtype=np.float32)
    m = float(np.max(np.abs(signal))) if signal.size else 0.0
    if m < EPS:
        return signal.copy()
    return (signal * (peak / m)).astype(np.float32)


def rms_normalize(signal: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS 归一化到目标 dBFS，并做峰值保护避免削波。"""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal.copy()
    rms = float(np.sqrt(np.mean(signal**2)))
    if rms < EPS:
        return signal.copy()
    gain = db_to_amp(target_db) / rms
    out = signal * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out *= 0.99 / peak
    return out.astype(np.float32)


__all__ = ["amp_to_db", "db_to_amp", "peak_normalize", "rms_normalize"]
