"""DDSP 合成器：谐波加性振荡器 + 时变子带增益滤波噪声。

提供 numpy 版（离线渲染/流式仿真，对应 ESP32 C 实现）与 torch 版（可微，用于
多尺度 STFT 损失）。两者与 analysis 的标签约定一致：
  - 谐波幅度 = 时域正弦幅度；
  - 噪声子带 = 该带噪声的时域 RMS（对单位 RMS 带通噪声乘以逐帧增益实现）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# 控制率 → 音频率 的线性插值
# --------------------------------------------------------------------------- #
def _upsample_np(values: np.ndarray, hop_size: int, n_samples: int) -> np.ndarray:
    """(T,) 或 (T,C) 帧级序列线性插值到 (n_samples,) 或 (n_samples,C)。"""
    values = np.asarray(values, dtype=np.float64)
    T = values.shape[0]
    frame_pos = np.arange(T) * hop_size
    sample_pos = np.arange(n_samples)
    if values.ndim == 1:
        return np.interp(sample_pos, frame_pos, values)
    out = np.empty((n_samples, values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(sample_pos, frame_pos, values[:, c])
    return out


def _band_edges_to_hz(band_edges: np.ndarray, sample_rate: int, fft_size: int) -> np.ndarray:
    """把（分析用 fft_size 的）bin 边界换算为 Hz 边界。"""
    return np.asarray(band_edges, dtype=np.float64) * (sample_rate / fft_size)


# --------------------------------------------------------------------------- #
# numpy 版合成
# --------------------------------------------------------------------------- #
def harmonic_synth_np(
    f0_hz: np.ndarray,
    harmonic_amp: np.ndarray,
    sample_rate: int,
    hop_size: int,
    n_samples: Optional[int] = None,
) -> np.ndarray:
    """加性谐波合成。f0_hz:(T,), harmonic_amp:(T,K) -> 音频 (n_samples,)。"""
    f0_hz = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
    harmonic_amp = np.asarray(harmonic_amp, dtype=np.float64)
    T, K = harmonic_amp.shape
    if n_samples is None:
        n_samples = T * hop_size

    f0_up = _upsample_np(f0_hz, hop_size, n_samples)          # (n,)
    amp_up = _upsample_np(harmonic_amp, hop_size, n_samples)  # (n,K)

    phase = np.cumsum(2.0 * np.pi * f0_up / sample_rate)      # 基频瞬时相位
    nyquist = sample_rate / 2.0
    y = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, K + 1):
        mask = (k * f0_up) < nyquist
        y += amp_up[:, k - 1] * np.sin(k * phase) * mask
    return y.astype(np.float32)


def filtered_noise_np(
    noise_band: np.ndarray,
    sample_rate: int,
    hop_size: int,
    fft_size: int,
    band_edges: np.ndarray,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """时变子带增益滤波噪声。noise_band:(T,B) 为各带时域 RMS -> 音频 (n_samples,)。

    做法：对整段单位 RMS 的带通白噪声乘以逐帧上采样的带增益，无接缝、RMS 可控。
    """
    noise_band = np.asarray(noise_band, dtype=np.float64)
    T, B = noise_band.shape
    if n_samples is None:
        n_samples = T * hop_size

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    hz_edges = _band_edges_to_hz(band_edges, sample_rate, fft_size)

    gains_up = _upsample_np(noise_band, hop_size, n_samples)  # (n,B)
    out = np.zeros(n_samples, dtype=np.float64)
    for b in range(B):
        lo, hi = hz_edges[b], hz_edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band_sig = np.fft.irfft(spec * mask, n=n_samples)
        rms = np.sqrt(np.mean(band_sig**2)) + _EPS
        out += (band_sig / rms) * gains_up[:, b]              # 单位 RMS × 逐帧增益
    return out.astype(np.float32)


def ddsp_synth_np(
    f0_hz: np.ndarray,
    harmonic_amp: np.ndarray,
    noise_band: np.ndarray,
    sample_rate: int,
    hop_size: int,
    fft_size: int,
    band_edges: np.ndarray,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """谐波 + 噪声完整合成（numpy）。"""
    T = harmonic_amp.shape[0]
    if n_samples is None:
        n_samples = T * hop_size
    y_h = harmonic_synth_np(f0_hz, harmonic_amp, sample_rate, hop_size, n_samples)
    y_n = filtered_noise_np(
        noise_band, sample_rate, hop_size, fft_size, band_edges, n_samples, seed
    )
    return (y_h + y_n).astype(np.float32)


# --------------------------------------------------------------------------- #
# torch 版合成（可微，用于多尺度 STFT 损失）
# --------------------------------------------------------------------------- #
def _upsample_torch(x, n_samples: int):
    """(B,T,C) -> (B,n_samples,C) 线性插值。"""
    import torch.nn.functional as F

    x = x.transpose(1, 2)                       # (B,C,T)
    x = F.interpolate(x, size=n_samples, mode="linear", align_corners=True)
    return x.transpose(1, 2)                    # (B,n,C)


def harmonic_synth_torch(f0_hz, harmonic_amp, sample_rate: int, hop_size: int,
                         n_samples: Optional[int] = None):
    """可微谐波合成。f0_hz:(B,T), harmonic_amp:(B,T,K) -> (B,n_samples)。"""
    import torch

    B, T, K = harmonic_amp.shape
    if n_samples is None:
        n_samples = T * hop_size

    f0_up = _upsample_torch(f0_hz.unsqueeze(-1), n_samples).squeeze(-1)   # (B,n)
    amp_up = _upsample_torch(harmonic_amp, n_samples)                     # (B,n,K)

    phase = torch.cumsum(2.0 * np.pi * f0_up / sample_rate, dim=-1)       # (B,n)
    k = torch.arange(1, K + 1, device=f0_hz.device, dtype=phase.dtype)    # (K,)
    phases_k = phase.unsqueeze(-1) * k                                    # (B,n,K)
    nyquist = sample_rate / 2.0
    mask = (f0_up.unsqueeze(-1) * k) < nyquist                            # (B,n,K)
    y = (amp_up * torch.sin(phases_k) * mask).sum(dim=-1)                 # (B,n)
    return y


def filtered_noise_torch(noise_band, sample_rate: int, hop_size: int, fft_size: int,
                         band_edges, n_samples: Optional[int] = None,
                         generator=None):
    """可微子带噪声合成。noise_band:(B,T,Bands) -> (B,n_samples)。

    随机噪声本身不可微（detach），梯度经由各带增益回传。
    """
    import torch

    Bsz, T, Bands = noise_band.shape
    if n_samples is None:
        n_samples = T * hop_size
    device, dtype = noise_band.device, noise_band.dtype

    white = torch.randn(Bsz, n_samples, device=device, dtype=dtype, generator=generator)
    spec = torch.fft.rfft(white, dim=-1)                                  # (B,F)
    freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate, device=device)
    hz_edges = np.asarray(band_edges, dtype=np.float64) * (sample_rate / fft_size)

    gains_up = _upsample_torch(noise_band, n_samples)                     # (B,n,Bands)
    out = torch.zeros(Bsz, n_samples, device=device, dtype=dtype)
    for b in range(Bands):
        lo, hi = float(hz_edges[b]), float(hz_edges[b + 1])
        mask = ((freqs >= lo) & (freqs < hi)).to(dtype)
        if mask.sum() == 0:
            continue
        band_sig = torch.fft.irfft(spec * mask, n=n_samples, dim=-1)      # (B,n)
        rms = torch.sqrt(torch.mean(band_sig**2, dim=-1, keepdim=True)) + _EPS
        out = out + (band_sig / rms).detach() * gains_up[:, :, b]
    return out


def ddsp_synth_torch(f0_hz, harmonic_amp, noise_band, sample_rate: int, hop_size: int,
                     fft_size: int, band_edges, n_samples: Optional[int] = None):
    """谐波 + 噪声完整合成（torch，可微）。"""
    T = harmonic_amp.shape[1]
    if n_samples is None:
        n_samples = T * hop_size
    y_h = harmonic_synth_torch(f0_hz, harmonic_amp, sample_rate, hop_size, n_samples)
    y_n = filtered_noise_torch(noise_band, sample_rate, hop_size, fft_size, band_edges, n_samples)
    return y_h + y_n


__all__ = [
    "harmonic_synth_np",
    "filtered_noise_np",
    "ddsp_synth_np",
    "harmonic_synth_torch",
    "filtered_noise_torch",
    "ddsp_synth_torch",
]
