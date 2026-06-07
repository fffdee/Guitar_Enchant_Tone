"""MFCC 及频谱特征基元（纯 numpy，HTK 风格 Mel 刻度）。

与未来 C 实现保持一致：FFT(补零) → 功率谱 → 三角 Mel 滤波 → log → DCT-II。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    fft_size: int,
    n_mels: int = 26,
    fmin: float = 30.0,
    fmax: float | None = None,
) -> np.ndarray:
    """构造三角 Mel 滤波器组，形状 (n_mels, fft_size//2 + 1)。"""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_bins = fft_size // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        if f_right <= f_left:
            continue
        left = (fft_freqs - f_left) / max(f_center - f_left, _EPS)
        right = (f_right - fft_freqs) / max(f_right - f_center, _EPS)
        fb[m - 1] = np.clip(np.minimum(left, right), 0.0, None)
    return fb.astype(np.float32)


def dct_ii(x: np.ndarray, n_out: int) -> np.ndarray:
    """正交归一化 DCT-II，对最后一维做变换，返回前 n_out 个系数。

    x: (..., n_in) -> (..., n_out)
    """
    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[-1]
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    basis = np.cos(np.pi * (2.0 * n + 1.0) * k / (2.0 * n_in))  # (n_out, n_in)
    scale = np.full((n_out, 1), np.sqrt(2.0 / n_in))
    scale[0, 0] = np.sqrt(1.0 / n_in)
    basis = basis * scale
    return (x @ basis.T).astype(np.float32)


def power_spectrum(frame: np.ndarray, fft_size: int, window: np.ndarray | None = None) -> np.ndarray:
    """单帧功率谱 |FFT|^2，长度 fft_size//2 + 1。"""
    frame = np.asarray(frame, dtype=np.float64)
    if window is not None:
        frame = frame[: len(window)] * window
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))
    else:
        frame = frame[:fft_size]
    spec = np.fft.rfft(frame, n=fft_size)
    return (spec.real**2 + spec.imag**2).astype(np.float64)


def mfcc_frame(
    power_spec: np.ndarray,
    mel_fb: np.ndarray,
    n_mfcc: int = 13,
) -> np.ndarray:
    """由功率谱计算 MFCC（前 n_mfcc 个系数）。"""
    mel_energy = mel_fb.astype(np.float64) @ power_spec
    log_mel = np.log(np.maximum(mel_energy, _EPS))
    return dct_ii(log_mel, n_mfcc)


__all__ = [
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "dct_ii",
    "power_spectrum",
    "mfcc_frame",
]
