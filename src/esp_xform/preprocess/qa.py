"""数据质检：单文件健康度 + 源/目标配对一致性。"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..audio.normalize import amp_to_db
from .align import estimate_lag
from .prepare import _frame_rms_db


def analyze_audio(x: np.ndarray, sample_rate: int, silence_db: float = -50.0) -> Dict:
    """返回单文件健康度指标。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = len(x)
    if n == 0:
        return {"n_samples": 0, "duration_s": 0.0, "empty": True}

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    db, _ = _frame_rms_db(x, max(1, int(0.02 * sample_rate)), max(1, int(0.01 * sample_rate)))
    silence_ratio = float(np.mean(db < silence_db)) if db.size else 1.0
    return {
        "n_samples": int(n),
        "duration_s": round(n / sample_rate, 3),
        "peak": round(peak, 4),
        "rms_db": round(float(amp_to_db(rms)), 2),
        "clip_ratio": round(float(np.mean(np.abs(x) >= 0.999)), 6),
        "silence_ratio": round(silence_ratio, 4),
        "dc_offset": round(float(np.mean(x)), 6),
        "n_nan": int(np.isnan(x).sum()),
        "n_inf": int(np.isinf(x).sum()),
        "empty": False,
    }


def qa_pair(
    source: np.ndarray, target: np.ndarray, sample_rate: int,
    max_lag_ms: float = 100.0, lag_warn_ms: float = 5.0,
) -> Dict:
    """源/目标配对质检：长度差、估计延迟、各自健康度与问题标记。"""
    src_a = analyze_audio(source, sample_rate)
    tgt_a = analyze_audio(target, sample_rate)
    max_lag = int(max_lag_ms * sample_rate / 1000)
    lag = estimate_lag(source, target, max_lag) if min(len(source), len(target)) > 0 else 0
    lag_ms = 1000.0 * lag / sample_rate

    len_diff = abs(len(source) - len(target))
    issues = []
    if src_a.get("empty") or tgt_a.get("empty"):
        issues.append("empty_audio")
    if abs(lag_ms) > lag_warn_ms:
        issues.append(f"misaligned({lag_ms:.1f}ms)")
    if len_diff > int(0.05 * sample_rate):
        issues.append(f"length_mismatch({len_diff}smp)")
    for tag, a in (("src", src_a), ("tgt", tgt_a)):
        if a.get("clip_ratio", 0) > 1e-3:
            issues.append(f"{tag}_clipping")
        if a.get("n_nan", 0) or a.get("n_inf", 0):
            issues.append(f"{tag}_nan_inf")
        if a.get("silence_ratio", 0) > 0.6:
            issues.append(f"{tag}_mostly_silent")

    return {
        "source": src_a,
        "target": tgt_a,
        "lag_samples": int(lag),
        "lag_ms": round(lag_ms, 2),
        "length_diff_samples": int(len_diff),
        "issues": issues,
        "ok": len(issues) == 0,
    }


__all__ = ["analyze_audio", "qa_pair"]
