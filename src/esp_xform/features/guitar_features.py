"""吉他侧 20 维逐帧特征装配。

特征布局（与 readme 方案一致，便于 C 端对齐）：
  [0]      log_f0           对数基频
  [1]      loudness_db      响度（RMS→dB）
  [2:15]   mfcc[0:13]       13 维 MFCC
  [15]     spectral_centroid 频谱质心（归一化到 [0,1]）
  [16]     spectral_rolloff  85% 能量滚降点（归一化）
  [17]     voiced_flag       有声标志 0/1
  [18]     zcr               过零率
  [19]     spectral_flux     频谱通量（相邻帧幅度谱正向差）

输出为“原始特征”（未标准化）；训练前再用全局 mean/std 标准化。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..audio.framing import frame_signal, hann_window
from ..audio.normalize import amp_to_db
from .f0_yin import yin_f0_frames
from .mfcc import mel_filterbank, dct_ii

FEATURE_LAYOUT = (
    ["log_f0", "loudness_db"]
    + [f"mfcc_{i}" for i in range(13)]
    + ["spectral_centroid", "spectral_rolloff", "voiced_flag", "zcr", "spectral_flux"]
)

_EPS = 1e-10


class GuitarFeatureExtractor:
    """从单声道音频提取逐帧 20 维特征，并返回合成所需的对齐量（f0/响度/有声）。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        a, f = cfg.audio, cfg.feature
        self.sr = a.sample_rate
        self.frame_size = a.frame_size
        self.hop_size = a.hop_size
        self.fft_size = a.fft_size
        self.n_mfcc = f.n_mfcc
        self.f0_min = f.f0_min
        self.f0_max = f.f0_max
        self.rolloff_percent = f.rolloff_percent
        self.silence_db = f.silence_db
        self.feature_dim = f.feature_dim

        self.window = hann_window(self.frame_size)
        self.mel_fb = mel_filterbank(self.sr, self.fft_size, f.n_mels, f.fmin, f.fmax)
        self.fft_freqs = np.linspace(0.0, self.sr / 2.0, self.fft_size // 2 + 1)
        self.nyquist = self.sr / 2.0

    def extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        frames = frame_signal(audio, self.frame_size, self.hop_size)  # (T, frame)
        n = frames.shape[0]
        if n == 0:
            return self._empty()

        # --- 基频（用未加窗帧）---
        f0, voiced, _aper = yin_f0_frames(
            frames, self.sr, self.f0_min, self.f0_max
        )

        # --- 频谱（加窗 + 补零 FFT，批量）---
        win_frames = frames * self.window[None, :]
        spec = np.fft.rfft(win_frames, n=self.fft_size, axis=1)
        power = (spec.real**2 + spec.imag**2)            # (T, n_bins)
        mag = np.sqrt(power)

        # --- 响度（原始帧 RMS → dB）---
        rms = np.sqrt(np.mean(frames**2, axis=1) + _EPS)
        loudness_db = amp_to_db(rms).astype(np.float32)
        silence = loudness_db < self.silence_db

        # --- MFCC ---
        mel_energy = power @ self.mel_fb.T.astype(np.float64)   # (T, n_mels)
        log_mel = np.log(np.maximum(mel_energy, _EPS))
        mfcc = dct_ii(log_mel, self.n_mfcc)                      # (T, n_mfcc)

        # --- 质心 / 滚降 ---
        mag_sum = np.sum(mag, axis=1) + _EPS
        centroid = (mag @ self.fft_freqs) / mag_sum
        centroid_norm = (centroid / self.nyquist).astype(np.float32)

        cum_energy = np.cumsum(power, axis=1)
        total_energy = cum_energy[:, -1:] + _EPS
        thresh = self.rolloff_percent * total_energy
        rolloff_idx = np.argmax(cum_energy >= thresh, axis=1)
        rolloff_norm = (self.fft_freqs[rolloff_idx] / self.nyquist).astype(np.float32)

        # --- 过零率 ---
        signs = np.sign(frames)
        signs[signs == 0] = 1.0
        zcr = (np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)).astype(np.float32)

        # --- 频谱通量（相邻帧正向差 L2，归一化）---
        flux = np.zeros(n, dtype=np.float32)
        if n > 1:
            diff = np.diff(mag, axis=0)
            diff[diff < 0] = 0.0
            flux[1:] = np.sqrt(np.sum(diff**2, axis=1)) / np.sqrt(mag.shape[1])

        # --- log_f0：静音/无声以 f0_min 作为下限 ---
        f0_safe = np.where(f0 > 0, f0, self.f0_min)
        log_f0 = np.log(np.clip(f0_safe, self.f0_min, self.f0_max)).astype(np.float32)

        voiced = np.where(silence, 0.0, voiced).astype(np.float32)

        feats = np.concatenate(
            [
                log_f0[:, None],
                loudness_db[:, None],
                mfcc,
                centroid_norm[:, None],
                rolloff_norm[:, None],
                voiced[:, None],
                zcr[:, None],
                flux[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        feats = self._fit_dim(feats)

        return {
            "features": feats,                       # (T, feature_dim)
            "f0_hz": np.where(silence, 0.0, f0).astype(np.float32),
            "loudness_db": loudness_db,
            "voiced": voiced,
            "n_frames": np.int64(n),
        }

    def _fit_dim(self, feats: np.ndarray) -> np.ndarray:
        d = feats.shape[1]
        if d == self.feature_dim:
            return feats
        if d < self.feature_dim:                     # 维度不足补零
            pad = np.zeros((feats.shape[0], self.feature_dim - d), dtype=np.float32)
            return np.concatenate([feats, pad], axis=1)
        return feats[:, : self.feature_dim]          # 维度过多截断

    def _empty(self) -> Dict[str, np.ndarray]:
        return {
            "features": np.zeros((0, self.feature_dim), dtype=np.float32),
            "f0_hz": np.zeros((0,), dtype=np.float32),
            "loudness_db": np.zeros((0,), dtype=np.float32),
            "voiced": np.zeros((0,), dtype=np.float32),
            "n_frames": np.int64(0),
        }


__all__ = ["GuitarFeatureExtractor", "FEATURE_LAYOUT"]
