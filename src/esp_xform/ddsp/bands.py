"""噪声子带划分（Mel 间隔），分析与合成共用，保证约定一致。"""

from __future__ import annotations

import numpy as np

from ..features.mfcc import hz_to_mel, mel_to_hz


def make_band_edges(
    sample_rate: int,
    fft_size: int,
    n_bands: int,
    fmin: float = 40.0,
    fmax: float | None = None,
) -> np.ndarray:
    """返回长度 n_bands+1 的 bin 索引边界（Mel 间隔），用于把谱划分为 n_bands 个子带。"""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_bins = fft_size // 2 + 1
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_bands + 1)
    hz_edges = mel_to_hz(mel_edges)
    bin_edges = np.round(hz_edges / (sample_rate / 2.0) * (n_bins - 1)).astype(int)
    bin_edges = np.clip(bin_edges, 0, n_bins - 1)
    # 保证单调且每带至少 1 个 bin
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = min(bin_edges[i - 1] + 1, n_bins - 1)
    return bin_edges


__all__ = ["make_band_edges"]
