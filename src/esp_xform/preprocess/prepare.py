"""音频规整：单声道化、重采样到 48k、幅度归一化、去首尾静音、定长。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..audio.normalize import amp_to_db, peak_normalize, rms_normalize
from ..audio.resample import resample_to


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    return x.reshape(-1)


def normalize_audio(x: np.ndarray, mode: str = "peak", target: float = 0.9) -> np.ndarray:
    """mode: 'peak'（目标峰值）/ 'rms'（目标 dBFS）/ 'none'。"""
    if mode == "peak":
        return peak_normalize(x, target)
    if mode == "rms":
        return rms_normalize(x, target if target < 0 else -20.0)
    return np.asarray(x, dtype=np.float32)


def _frame_rms_db(x: np.ndarray, frame: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < frame:
        return np.array([amp_to_db(np.sqrt(np.mean(x**2) + 1e-12))]), np.array([0])
    starts = np.arange(0, n - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(x[s : s + frame] ** 2) + 1e-12) for s in starts])
    return amp_to_db(rms), starts


def trim_silence(
    x: np.ndarray, sample_rate: int, thresh_db: float = -50.0,
    frame_ms: float = 20.0, hop_ms: float = 10.0, pad_ms: float = 30.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """去除首尾静音，返回 (裁剪后音频, (start,end) 样点)。全静音则原样返回。"""
    x = np.asarray(x, dtype=np.float32)
    frame = max(1, int(frame_ms * sample_rate / 1000))
    hop = max(1, int(hop_ms * sample_rate / 1000))
    db, starts = _frame_rms_db(x, frame, hop)
    active = np.where(db >= thresh_db)[0]
    if active.size == 0:
        return x.copy(), (0, len(x))
    pad = int(pad_ms * sample_rate / 1000)
    s = max(0, int(starts[active[0]]) - pad)
    e = min(len(x), int(starts[active[-1]]) + frame + pad)
    return x[s:e].copy(), (s, e)


def fix_length(x: np.ndarray, length: int) -> np.ndarray:
    """定长：不足补零，超长截断。"""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == length:
        return x.copy()
    if len(x) < length:
        return np.pad(x, (0, length - len(x)))
    return x[:length].copy()


def prepare_wav(
    audio: np.ndarray, orig_sr: int, target_sr: int = 48000,
    mono: bool = True, normalize: str = "peak", norm_target: float = 0.9,
    trim: bool = False, trim_thresh_db: float = -50.0,
) -> np.ndarray:
    """完整规整流水线：→单声道→重采样→(去静音)→归一化。"""
    x = audio
    if mono:
        x = to_mono(x)
    if orig_sr != target_sr:
        x = resample_to(x, orig_sr, target_sr)
    if trim:
        x, _ = trim_silence(x, target_sr, trim_thresh_db)
    x = normalize_audio(x, normalize, norm_target)
    return x.astype(np.float32)


__all__ = ["to_mono", "normalize_audio", "trim_silence", "fix_length", "prepare_wav"]
