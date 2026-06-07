"""损失函数：参数级 MAE（谐波/噪声）+ 可选多尺度 STFT 损失。"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

_EPS = 1e-7


def param_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_harmonics: int,
    noise_weight: float = 0.5,
) -> torch.Tensor:
    """参数损失：L = MAE(谐波) + noise_weight·MAE(噪声)。

    pred/target: (B, T, output_dim)，前 n_harmonics 维为谐波，其余为噪声。
    """
    harm_pred, noise_pred = pred[..., :n_harmonics], pred[..., n_harmonics:]
    harm_tgt, noise_tgt = target[..., :n_harmonics], target[..., n_harmonics:]
    l_harm = F.l1_loss(harm_pred, harm_tgt)
    l_noise = F.l1_loss(noise_pred, noise_tgt)
    return l_harm + noise_weight * l_noise


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, center=True, return_complex=True,
    )
    return spec.abs()


def multi_scale_stft_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128),
    hop_ratio: float = 0.25,
) -> torch.Tensor:
    """多尺度 STFT 损失：各尺度上 |STFT| 的 L1 + log|STFT| 的 L1 之和。

    y_pred/y_true: (B, n_samples)。对幅度与对数幅度同时约束，兼顾强弱成分。
    """
    n = min(y_pred.shape[-1], y_true.shape[-1])
    y_pred, y_true = y_pred[..., :n], y_true[..., :n]
    total = y_pred.new_zeros(())
    for n_fft in fft_sizes:
        hop = max(1, int(n_fft * hop_ratio))
        s_pred = _stft_mag(y_pred, n_fft, hop)
        s_true = _stft_mag(y_true, n_fft, hop)
        lin = F.l1_loss(s_pred, s_true)
        log = F.l1_loss(torch.log(s_pred + _EPS), torch.log(s_true + _EPS))
        total = total + lin + log
    return total / len(fft_sizes)


__all__ = ["param_loss", "multi_scale_stft_loss"]
