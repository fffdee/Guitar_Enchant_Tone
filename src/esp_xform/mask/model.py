"""MaskNet：条件谱掩码 CNN（FiLM + 乐器嵌入，三输出头）。对应 IMPLEMENTATION.md §3。

主干 Conv1D 96→128→128→96 + FiLM×2；三头：
  幅度掩码(sigmoid×Gmax) / 相位残差(tanh×Δφmax) / 噪声带增益(softplus)。
支持因果 padding（实时流式部署只用过去帧）。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """逐通道线性调制 y = gamma*x + beta（按 §3.4 实现，不含 1+gamma 偏置）。"""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:  # x:[B,C,T]
        gb = self.fc(cond)                                   # [B, 2C]
        gamma, beta = gb[:, : self.channels], gb[:, self.channels:]
        return x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)


def _causal_conv(x: torch.Tensor, conv: nn.Conv1d, dilation: int, kernel: int) -> torch.Tensor:
    """因果 1D 卷积：仅左侧补零 dilation*(kernel-1)，conv 自身 padding=0。"""
    pad = dilation * (kernel - 1)
    return conv(F.pad(x, (pad, 0)))


class MaskNet(nn.Module):
    def __init__(self, cfg, num_instruments: int) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        n_mels = cfg.stft.n_mels
        C = m.hidden
        k = m.kernel
        self.causal = m.causal
        self.kernel = k
        self.dil2 = m.dilation2
        self.gmax = m.gmax
        self.dphi_max = m.dphi_max
        self.n_mels = n_mels
        self.phase_bands = m.phase_bands
        self.noise_bands = m.noise_bands

        self.emb = nn.Embedding(num_instruments, m.emb_dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.use_artic = m.n_artic > 0
        self.aemb = nn.Embedding(m.n_artic, m.artic_dim) if self.use_artic else None
        cond_dim = m.emb_dim + (m.artic_dim if self.use_artic else 0)

        # 主干：非因果用 same padding，因果用手动左 padding(forward 处理)
        p1 = 0 if self.causal else (k - 1) // 2
        p2 = 0 if self.causal else self.dil2 * (k - 1) // 2
        self.c1 = nn.Conv1d(n_mels, C, k, padding=p1)
        self.f1 = FiLM(cond_dim, C)
        self.c2 = nn.Conv1d(C, C, k, padding=p2, dilation=self.dil2)
        self.f2 = FiLM(cond_dim, C)
        self.c3 = nn.Conv1d(C, n_mels, k, padding=p1)

        self.head_mag = nn.Conv1d(n_mels, n_mels, 1)
        self.head_phase = nn.Conv1d(n_mels, m.phase_bands, 1)
        self.head_noise = nn.Conv1d(n_mels, m.noise_bands, 1)
        self.act = nn.ReLU()
        self.num_instruments = num_instruments

    # ------------------------------------------------------------------ #
    def cond_vector(self, inst_id: torch.Tensor, artic_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.emb(inst_id)
        if self.use_artic and artic_id is not None:
            cond = torch.cat([cond, self.aemb(artic_id)], dim=-1)
        return cond

    def _conv(self, x: torch.Tensor, conv: nn.Conv1d, dilation: int) -> torch.Tensor:
        if self.causal:
            return _causal_conv(x, conv, dilation, self.kernel)
        return conv(x)

    def forward(self, x: torch.Tensor, inst_id: torch.Tensor,
                artic_id: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None):
        """x:[B,n_mels,T] -> (mask[B,n_mels,T], dphi[B,P,T], noise[B,B_bands,T])。"""
        if cond is None:
            cond = self.cond_vector(inst_id, artic_id)
        h = self.f1(self.act(self._conv(x, self.c1, 1)), cond)
        h = self.f2(self.act(self._conv(h, self.c2, self.dil2)), cond)
        h = self.act(self._conv(h, self.c3, 1))
        mask = torch.sigmoid(self.head_mag(h)) * self.gmax
        dphi = torch.tanh(self.head_phase(h)) * self.dphi_max
        noise = F.softplus(self.head_noise(h))
        return mask, dphi, noise

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # 供导出器使用：主干+三头的卷积/FiLM 顺序（与 C 计算图遍历顺序一致）
    def ordered_modules(self) -> List[nn.Module]:
        return [self.c1, self.f1, self.c2, self.f2, self.c3,
                self.head_mag, self.head_phase, self.head_noise]


__all__ = ["FiLM", "MaskNet"]
