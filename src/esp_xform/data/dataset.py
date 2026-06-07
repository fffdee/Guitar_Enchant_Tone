"""数据集：加载缓存的 (features, labels, f0) 样本，按固定长度序列窗供训练。

缓存格式（由 scripts/extract_*.py 生成的 .npz）：
  features:(T,feature_dim) float32, labels:(T,output_dim) float32,
  f0_hz:(T,) float32, instrument_id:() int64, sample_id:str
特征在此做全局 mean/std 标准化；标签保持原始物理量纲，使合成器可直接使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_example(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"].astype(np.float32),
        "labels": data["labels"].astype(np.float32),
        "f0_hz": data["f0_hz"].astype(np.float32),
        "instrument_id": np.int64(data["instrument_id"]),
        "sample_id": str(data["sample_id"]) if "sample_id" in data else Path(path).stem,
    }


def compute_feature_stats(
    example_paths: Sequence[str | Path],
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """在给定样本（通常为训练集）上统计逐维特征 mean/std。"""
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for p in example_paths:
        feats = np.load(p, allow_pickle=True)["features"].astype(np.float64)
        if feats.size == 0:
            continue
        total += feats.sum(axis=0)
        total_sq += (feats**2).sum(axis=0)
        count += feats.shape[0]
    if count == 0:
        return np.zeros(feature_dim, np.float32), np.ones(feature_dim, np.float32)
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


class DDSPFrameDataset(Dataset):
    def __init__(
        self,
        example_paths: Sequence[str | Path],
        sequence_length: int,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        window_stride: Optional[int] = None,
    ) -> None:
        self.L = int(sequence_length)
        self.stride = int(window_stride) if window_stride else max(1, self.L // 2)
        self.feature_mean = None if feature_mean is None else feature_mean.astype(np.float32)
        self.feature_std = None if feature_std is None else feature_std.astype(np.float32)

        self.examples: List[Dict[str, np.ndarray]] = []
        for p in example_paths:
            ex = _load_example(p)
            if ex["features"].shape[0] > 0:
                self.examples.append(ex)

        # 预建窗口索引 (example_idx, start_frame)
        self.index: List[Tuple[int, int]] = []
        for ei, ex in enumerate(self.examples):
            T = ex["features"].shape[0]
            if T < self.L:
                self.index.append((ei, 0))           # 不足一窗 → 末尾补零
                continue
            starts = list(range(0, T - self.L + 1, self.stride))
            if starts[-1] != T - self.L:
                starts.append(T - self.L)            # 补一个贴尾窗
            self.index.extend((ei, s) for s in starts)

    def __len__(self) -> int:
        return len(self.index)

    def _standardize(self, feats: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return feats
        return (feats - self.feature_mean) / (self.feature_std + 1e-7)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, s = self.index[i]
        ex = self.examples[ei]
        L = self.L
        feats = ex["features"][s : s + L]
        labels = ex["labels"][s : s + L]
        f0 = ex["f0_hz"][s : s + L]

        if feats.shape[0] < L:                       # 末尾补零到定长
            pad = L - feats.shape[0]
            feats = np.pad(feats, ((0, pad), (0, 0)))
            labels = np.pad(labels, ((0, pad), (0, 0)))
            f0 = np.pad(f0, (0, pad))

        feats = self._standardize(feats)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(feats)),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)),
            "f0_hz": torch.from_numpy(np.ascontiguousarray(f0)),
            "instrument_id": torch.tensor(int(ex["instrument_id"]), dtype=torch.long),
        }


def collate_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "f0_hz": torch.stack([b["f0_hz"] for b in batch]),
        "instrument_id": torch.stack([b["instrument_id"] for b in batch]),
    }


__all__ = ["DDSPFrameDataset", "collate_sequences", "compute_feature_stats"]
