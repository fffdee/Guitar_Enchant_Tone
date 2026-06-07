"""切段：定长滑窗 + 能量 VAD 静音切分。

用于把长录音切成训练友好的片段；保持源/目标用相同切点即可维持对齐。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .prepare import _frame_rms_db


def segment_fixed(
    x: np.ndarray, segment_samples: int, hop_samples: int | None = None, drop_last: bool = False
) -> List[Tuple[int, int]]:
    """定长切段，返回 [(start, end), ...]（样点）。hop 缺省=segment（不重叠）。"""
    x = np.asarray(x)
    n = len(x)
    hop = hop_samples or segment_samples
    spans: List[Tuple[int, int]] = []
    s = 0
    while s < n:
        e = s + segment_samples
        if e > n:
            if drop_last:
                break
            e = n
        spans.append((s, e))
        if e == n:
            break
        s += hop
    return spans


def segment_by_silence(
    x: np.ndarray, sample_rate: int, thresh_db: float = -45.0,
    frame_ms: float = 20.0, hop_ms: float = 10.0,
    min_silence_ms: float = 200.0, min_segment_ms: float = 300.0, pad_ms: float = 30.0,
) -> List[Tuple[int, int]]:
    """基于能量的静音切分，返回有声片段 [(start, end), ...]（样点）。"""
    x = np.asarray(x, dtype=np.float32)
    frame = max(1, int(frame_ms * sample_rate / 1000))
    hop = max(1, int(hop_ms * sample_rate / 1000))
    db, starts = _frame_rms_db(x, frame, hop)
    active = db >= thresh_db
    if not np.any(active):
        return []

    # 合并被短静音分隔的有声区间
    min_sil_frames = int(min_silence_ms / hop_ms)
    idx = np.where(active)[0]
    groups: List[List[int]] = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - groups[-1][1] <= min_sil_frames:
            groups[-1][1] = i
        else:
            groups.append([i, i])

    pad = int(pad_ms * sample_rate / 1000)
    min_seg = int(min_segment_ms * sample_rate / 1000)
    spans: List[Tuple[int, int]] = []
    for g0, g1 in groups:
        s = max(0, int(starts[g0]) - pad)
        e = min(len(x), int(starts[g1]) + frame + pad)
        if e - s >= min_seg:
            spans.append((s, e))
    return spans


__all__ = ["segment_fixed", "segment_by_silence"]
