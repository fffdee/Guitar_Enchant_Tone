"""DDSP 标签分析：从目标乐器音频提取 30 维谐波幅度 + 10 维噪声子带 RMS。

约定（与 synthesizer 严格配对）：
  - 谐波幅度 A_k 为时域正弦幅度，由加窗谱峰换算：A_k = 2·|X_peak| / Σw
  - 噪声子带为该带内残差信号的时域 RMS（用 Parseval 由加窗谱反推，并补偿窗能量）
谐波定位使用与源吉他共享的逐帧 f0（平行语料同一 MIDI，音高一致）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..audio.framing import frame_signal, hann_window
from .bands import make_band_edges

_EPS = 1e-10


class DDSPAnalyzer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        a, d, f = cfg.audio, cfg.ddsp, cfg.feature
        self.sr = a.sample_rate
        self.frame_size = a.frame_size
        self.hop_size = a.hop_size
        self.fft_size = a.fft_size
        self.n_harmonics = d.n_harmonics
        self.n_bands = d.n_noise_bands
        self.nyquist = self.sr / 2.0

        self.window = hann_window(self.frame_size)
        self.win_sum = float(np.sum(self.window))
        self.win_sq_sum = float(np.sum(self.window**2))
        self.n_bins = self.fft_size // 2 + 1
        self.band_edges = make_band_edges(
            self.sr, self.fft_size, self.n_bands, fmin=max(20.0, f.fmin)
        )

    def analyze(self, target_audio: np.ndarray, f0_hz: np.ndarray) -> Dict[str, np.ndarray]:
        """返回 labels=(T,40)、harmonic=(T,30)、noise=(T,10)。f0_hz 为逐帧基频。"""
        frames = frame_signal(target_audio, self.frame_size, self.hop_size)
        T = frames.shape[0]
        f0 = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
        if len(f0) != T:                      # 对齐到较短者
            m = min(len(f0), T)
            frames, f0, T = frames[:m], f0[:m], m
        if T == 0:
            return self._empty()

        win_frames = frames * self.window[None, :]
        spec = np.fft.rfft(win_frames, n=self.fft_size, axis=1)
        power = spec.real**2 + spec.imag**2               # (T, n_bins)
        mag = np.sqrt(power)

        harmonic = np.zeros((T, self.n_harmonics), dtype=np.float64)
        harm_mask = np.zeros((T, self.n_bins), dtype=bool)
        bin_per_hz = self.fft_size / self.sr
        frame_idx = np.arange(T)
        voiced = f0 > 0

        for k in range(1, self.n_harmonics + 1):
            harm_freq = k * f0                            # (T,)
            valid = voiced & (harm_freq < self.nyquist)
            bink = np.clip(np.round(harm_freq * bin_per_hz).astype(int), 1, self.n_bins - 2)
            # 取主瓣峰值 bin 幅度（±1 内最大），换算时域正弦幅度 A=2|X|/Σw
            peak = np.maximum.reduce([
                mag[frame_idx, bink - 1],
                mag[frame_idx, bink],
                mag[frame_idx, bink + 1],
            ])
            amp_k = peak * 2.0 / max(self.win_sum, _EPS)
            harmonic[:, k - 1] = np.where(valid, amp_k, 0.0)
            # 标记谐波占用的 bin（用于残差噪声）
            for off in (-1, 0, 1):
                bb = np.clip(bink + off, 0, self.n_bins - 1)
                harm_mask[frame_idx[valid], bb[valid]] = True

        # --- 残差噪声：去掉谐波 bin 后按子带求时域 RMS ---
        residual_power = power.copy()
        residual_power[harm_mask] = 0.0
        residual_power[:, 0] = 0.0                        # 去直流

        noise = np.zeros((T, self.n_bands), dtype=np.float64)
        for b in range(self.n_bands):
            lo, hi = self.band_edges[b], self.band_edges[b + 1]
            if hi <= lo:
                hi = lo + 1
            band_pow = np.sum(residual_power[:, lo:hi], axis=1)
            # Parseval：时域能量 ≈ (2/N)Σ|Xw|^2；再除以 Σw^2 还原未加窗 RMS^2
            ms = (2.0 / self.fft_size) * band_pow / max(self.win_sq_sum, _EPS)
            noise[:, b] = np.sqrt(np.maximum(ms, 0.0))

        harmonic = harmonic.astype(np.float32)
        noise = noise.astype(np.float32)
        labels = np.concatenate([harmonic, noise], axis=1).astype(np.float32)
        return {"labels": labels, "harmonic": harmonic, "noise": noise, "n_frames": np.int64(T)}

    def _empty(self) -> Dict[str, np.ndarray]:
        return {
            "labels": np.zeros((0, self.n_harmonics + self.n_bands), dtype=np.float32),
            "harmonic": np.zeros((0, self.n_harmonics), dtype=np.float32),
            "noise": np.zeros((0, self.n_bands), dtype=np.float32),
            "n_frames": np.int64(0),
        }


__all__ = ["DDSPAnalyzer"]
