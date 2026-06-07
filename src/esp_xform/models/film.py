"""FiLM（Feature-wise Linear Modulation）调制层。

由乐器嵌入 e 生成逐通道的 γ、β，对特征图做 x' = (1+γ)·x + β。
线性层零初始化 → 训练初期为恒等变换，提升稳定性。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, embedding_dim: int, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.proj = nn.Linear(embedding_dim, 2 * num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)；emb: (B, E)
        gamma, beta = self.proj(emb).chunk(2, dim=-1)        # 各 (B, C)
        gamma = gamma.unsqueeze(-1)                          # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return (1.0 + gamma) * x + beta


__all__ = ["FiLM"]
