"""谱掩码方案混合损失（对应 IMPLEMENTATION.md §4.1）。

  L = w_mask·L_mask + w_logmel·L_logmel + w_mrstft·L_mrstft
      + w_cplx·L_cplx(启用相位) + w_adv·L_adv(后期, 见 train)
对抗损失在 train.py 单独处理（需判别器）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def mask_l1(mask_pred: torch.Tensor, mask_label: torch.Tensor) -> torch.Tensor:
    return (mask_pred - mask_label).abs().mean()


def logmel_l1(mask_pred: torch.Tensor, Xmel: torch.Tensor, Ymel: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    """对数梅尔重建 L1。Xmel/Ymel 为线性梅尔幅度 [B,M,T]。"""
    Yhat = mask_pred * Xmel
    return ((Yhat + eps).log() - (Ymel + eps).log()).abs().mean()


def multi_res_stft(y_pred: torch.Tensor, y_true: torch.Tensor,
                   ffts: List[int], hops: List[int], eps: float = 1e-5) -> torch.Tensor:
    """多分辨率 STFT 谱损失：各尺度线性幅度 L1 + 对数幅度 L1 之和。"""
    if y_pred.shape[-1] != y_true.shape[-1]:
        m = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred, y_true = y_pred[..., :m], y_true[..., :m]
    total = y_pred.new_zeros(())
    for n_fft, hop in zip(ffts, hops):
        win = torch.hann_window(n_fft, periodic=True, device=y_pred.device, dtype=y_pred.dtype)
        Sp = torch.stft(y_pred, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, center=True, return_complex=True).abs()
        St = torch.stft(y_true, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, center=True, return_complex=True).abs()
        total = total + (Sp - St).abs().mean()
        total = total + ((Sp + eps).log() - (St + eps).log()).abs().mean()
    return total / max(len(ffts), 1)


def complex_stft_loss(Xlin: torch.Tensor, phase: torch.Tensor, mask: torch.Tensor,
                      dphi: torch.Tensor, mel_inv: torch.Tensor, phase_inv: torch.Tensor,
                      target_mag: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
    """线性域复数谱损失：| S_target - (gain·e^{jΔφ}) ⊙ S_guitar |。相位只能在线性域(§13.2)。"""
    gain_lin = torch.einsum("rm,bmt->brt", mel_inv, mask)
    dphi_lin = torch.einsum("rp,bpt->brt", phase_inv, dphi)
    Mcplx = torch.polar(gain_lin, dphi_lin)
    S_guitar = torch.polar(Xlin, phase)
    S_target = torch.polar(target_mag, target_phase)
    pred = Mcplx * S_guitar
    return (S_target - pred).abs().mean()


__all__ = ["mask_l1", "logmel_l1", "multi_res_stft", "complex_stft_loss"]
