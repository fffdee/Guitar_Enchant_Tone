"""条件 DDSP 参数预测网络。

结构（对应 readme 5.3）：
  输入 = [20 维吉他特征 ⊕ 8 维乐器嵌入] = 28 通道
  膨胀 Conv1D 堆叠（same padding），在指定层后施加 FiLM 调制
  1×1 输出卷积 → 40 维（30 谐波幅度 + 10 噪声子带），softplus 保证非负
嵌入矩阵随网络一同训练；推理时切换乐器仅需更换嵌入向量。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM


class ConditionalDDSPNet(nn.Module):
    def __init__(self, cfg, num_instruments: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_dim = cfg.feature.feature_dim
        self.embedding_dim = cfg.model.embedding_dim
        self.n_harmonics = cfg.ddsp.n_harmonics
        self.n_noise_bands = cfg.ddsp.n_noise_bands
        self.output_dim = self.n_harmonics + self.n_noise_bands
        self.num_instruments = num_instruments

        self.embedding = nn.Embedding(num_instruments, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        channels = cfg.model.channels
        dilations = cfg.model.dilations
        kernel = cfg.model.kernel_size
        film_layers = set(cfg.model.film_layers)

        in_ch = self.feature_dim + self.embedding_dim
        self.convs = nn.ModuleList()
        self.films = nn.ModuleList()
        self.has_film = []
        prev = in_ch
        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            pad = dil * (kernel - 1) // 2                 # same padding（奇数核）
            self.convs.append(nn.Conv1d(prev, ch, kernel, dilation=dil, padding=pad))
            if i in film_layers:
                self.films.append(FiLM(self.embedding_dim, ch))
                self.has_film.append(True)
            else:
                self.films.append(None)  # type: ignore[arg-type]
                self.has_film.append(False)
            prev = ch
        self.head = nn.Conv1d(prev, self.output_dim, kernel_size=1)

    # ------------------------------------------------------------------ #
    def get_embedding(self, instrument_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(instrument_id)

    def forward(
        self,
        features: torch.Tensor,                          # (B, T, feature_dim)
        instrument_id: Optional[torch.Tensor] = None,    # (B,)
        embedding: Optional[torch.Tensor] = None,        # (B, E) 可直接给定
    ) -> torch.Tensor:
        """返回 DDSP 参数 (B, T, 40)，已 softplus 非负。"""
        B, T, _ = features.shape
        if embedding is None:
            if instrument_id is None:
                raise ValueError("instrument_id 与 embedding 至少提供其一")
            embedding = self.get_embedding(instrument_id)     # (B, E)

        emb_seq = embedding.unsqueeze(1).expand(B, T, self.embedding_dim)
        x = torch.cat([features, emb_seq], dim=-1)            # (B, T, 28)
        x = x.transpose(1, 2)                                  # (B, 28, T)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            if self.has_film[i]:
                x = self.films[i](x, embedding)
            x = F.relu(x)
        x = self.head(x)                                       # (B, 40, T)
        x = x.transpose(1, 2)                                  # (B, T, 40)
        return x                                               # 线性输出（标签非负，MAE 直接回归）

    def split_params(self, params: torch.Tensor):
        """拆分为 (harmonic_amp, noise_band)。"""
        return params[..., : self.n_harmonics], params[..., self.n_harmonics :]

    @staticmethod
    def to_synth_params(params: torch.Tensor) -> torch.Tensor:
        """供合成器使用：clamp 到非负（幅度/能量不能为负）。"""
        return torch.clamp(params, min=0.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["ConditionalDDSPNet"]
