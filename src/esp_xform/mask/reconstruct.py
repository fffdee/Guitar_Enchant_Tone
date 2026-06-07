"""音频重建链（对应 IMPLEMENTATION.md §5）。

谐波/音调：mel 掩码→线性增益→作用于吉他幅度→复用相位+有界残差→ISTFT+OLA。
噪声：噪声带增益→谱形→随机相位→ISTFT。
提供 torch 版（可微，训练用）与 numpy 版（离线渲染，镜像 C 端部署）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .audio import istft as np_istft

_EPS = 1e-8


class SpectralReconstructor(nn.Module):
    """持有展开矩阵与窗（buffer），由三分支输出 + 吉他幅度/相位重建波形。"""

    def __init__(self, cfg, mel_inv: np.ndarray, phase_inv: np.ndarray, noise_fb: np.ndarray) -> None:
        super().__init__()
        self.n_fft = cfg.stft.n_fft
        self.hop = cfg.stft.hop
        self.center = cfg.stft.center
        self.register_buffer("mel_inv", torch.tensor(np.asarray(mel_inv, dtype=np.float32)))      # [n_bins, M]
        self.register_buffer("phase_inv", torch.tensor(np.asarray(phase_inv, dtype=np.float32)))  # [n_bins, P]
        self.register_buffer("noise_fb", torch.tensor(np.asarray(noise_fb, dtype=np.float32)))    # [B_bands, n_bins]
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True))

    def _istft(self, Y: torch.Tensor, length: Optional[int]) -> torch.Tensor:
        return torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                           window=self.window, center=self.center, length=length)

    def forward(self, Xlin: torch.Tensor, phase: torch.Tensor, mask: torch.Tensor,
                dphi: torch.Tensor, noise: torch.Tensor, length: Optional[int] = None,
                add_noise: bool = True) -> torch.Tensor:
        """Xlin/phase:[B,n_bins,T]; mask:[B,M,T]; dphi:[B,P,T]; noise:[B,Bbands,T] -> y:[B,n_samples]。"""
        # 1) 梅尔掩码 -> 线性增益 -> 作用于吉他幅度
        gain_lin = torch.einsum("rm,bmt->brt", self.mel_inv, mask)        # [B,n_bins,T]
        Ylin = gain_lin * Xlin
        # 2) 相位 = 复用吉他相位 + 低分辨率残差展开
        dphi_lin = torch.einsum("rp,bpt->brt", self.phase_inv, dphi)
        ph = phase + dphi_lin
        Y = torch.polar(Ylin, ph)                                         # 复数 [B,n_bins,T]
        y = self._istft(Y, length)
        # 3) 噪声：带增益 -> 谱形 -> 随机相位 -> ISTFT
        if add_noise:
            noise_shape = torch.einsum("cn,bct->bnt", self.noise_fb, noise)  # [B,n_bins,T]
            rphase = (torch.rand_like(noise_shape) * 2.0 - 1.0) * np.pi
            N = torch.polar(noise_shape, rphase.detach())
            yn = self._istft(N, length if length is not None else y.shape[-1])
            if yn.shape[-1] != y.shape[-1]:
                m = min(yn.shape[-1], y.shape[-1])
                y = y[..., :m] + yn[..., :m]
            else:
                y = y + yn
        return y


# --------------------------------------------------------------------------- #
# numpy 离线重建（镜像 C 端；含增益平滑 + 噪声门 后处理）
# --------------------------------------------------------------------------- #
def _transient_shape(y: np.ndarray, amount: float,
                     a_fast: float = 0.3, a_slow: float = 0.05) -> np.ndarray:
    """轻量瞬态整形：快/慢包络差驱动起音增益（§5.10，默认低量）。"""
    ef = es = 0.0
    out = np.empty_like(y)
    for i in range(len(y)):
        ay = abs(float(y[i]))
        ef = a_fast * ay + (1.0 - a_fast) * ef
        es = a_slow * ay + (1.0 - a_slow) * es
        d = ef - es
        g = 1.0 + amount * (d if d > 0.0 else 0.0) / (es + 1e-6)
        out[i] = y[i] * g
    return out


def reconstruct_np(
    Xlin: np.ndarray,          # [n_bins, T] 吉他线性幅度
    phase: np.ndarray,         # [n_bins, T] 吉他相位
    mask: np.ndarray,          # [M, T]
    dphi: np.ndarray,          # [P, T]
    noise: np.ndarray,         # [Bbands, T]
    mel_inv: np.ndarray,       # [n_bins, M]
    phase_inv: np.ndarray,     # [n_bins, P]
    noise_fb: np.ndarray,      # [Bbands, n_bins]
    n_fft: int, hop: int, center: bool = True,
    smooth_a: float = 0.5, noise_gate_db: float = -60.0,
    add_noise: bool = True, seed: Optional[int] = 0,
    transient_amount: float = 0.0,
) -> np.ndarray:
    Xlin = np.asarray(Xlin, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    n_bins, T = Xlin.shape

    gain_lin = mel_inv.astype(np.float64) @ mask.astype(np.float64)       # [n_bins, T]
    # 帧间一阶平滑抑制掩码抖动
    if smooth_a > 0:
        for t in range(1, T):
            gain_lin[:, t] = smooth_a * gain_lin[:, t - 1] + (1.0 - smooth_a) * gain_lin[:, t]

    dphi_lin = phase_inv.astype(np.float64) @ dphi.astype(np.float64)     # [n_bins, T]
    Ylin = gain_lin * Xlin
    Y = (Ylin * np.exp(1j * (phase + dphi_lin))).T                       # [T, n_bins]
    y_tonal = np_istft(Y, n_fft, hop, center=center)

    if add_noise:
        rng = np.random.default_rng(seed)
        noise_shape = noise_fb.astype(np.float64).T @ noise.astype(np.float64)  # [n_bins, T]
        rphase = rng.uniform(-np.pi, np.pi, size=noise_shape.shape)
        N = (noise_shape * np.exp(1j * rphase)).T
        y_noise = np_istft(N, n_fft, hop, center=center)
        m = min(len(y_tonal), len(y_noise))
        y = y_tonal[:m] + y_noise[:m]
    else:
        y = y_tonal

    # 轻量瞬态整形 (默认关闭)
    if transient_amount > 0.0:
        y = _transient_shape(y, transient_amount)

    # 噪声门：逐帧输入能量低于阈值则该段静音
    frame_rms = np.sqrt(np.mean(Xlin**2, axis=0) + _EPS)
    gate = (20.0 * np.log10(frame_rms + _EPS)) > noise_gate_db
    gate_up = np.repeat(gate, hop)[: len(y)]
    if len(gate_up) < len(y):
        gate_up = np.pad(gate_up, (0, len(y) - len(gate_up)), constant_values=True)
    y = y * gate_up
    return y.astype(np.float32)


__all__ = ["SpectralReconstructor", "reconstruct_np"]
