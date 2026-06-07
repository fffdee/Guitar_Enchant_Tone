"""复音/单音判别（对应 §8）。由幅度谱估计显著谱峰数与谱平坦度，驱动双模切换。

C 端镜像见 MCU/Tinynn/operator/bnn_polyphony。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .audio import stft

_EPS = 1e-10


def analyze_mag(mag: np.ndarray, peak_rel_thresh: float = 0.1,
                peak_count_thresh: int = 6) -> Dict:
    """单帧幅度谱 [n_bins] -> {n_peaks, flatness, is_poly}。"""
    mag = np.asarray(mag, dtype=np.float64).reshape(-1)
    mx = float(mag.max()) if mag.size else 0.0
    thr = peak_rel_thresh * mx
    if mag.size > 2:
        center = mag[1:-1]
        peaks = (center > thr) & (center >= mag[:-2]) & (center > mag[2:])
        n_peaks = int(np.count_nonzero(peaks))
    else:
        n_peaks = 0
    p = mag ** 2 + _EPS
    gmean = np.exp(np.mean(np.log(p)))
    amean = np.mean(p)
    flatness = float(gmean / amean) if amean > _EPS else 0.0
    return {"n_peaks": n_peaks, "flatness": flatness, "is_poly": int(n_peaks >= peak_count_thresh)}


def analyze_audio(audio: np.ndarray, n_fft: int = 1024, hop: int = 256,
                  peak_rel_thresh: float = 0.1, peak_count_thresh: int = 6,
                  agg: str = "median") -> Dict:
    """整段音频逐帧判别后聚合：返回逐帧 is_poly 比例与聚合结论。"""
    X = np.abs(stft(audio, n_fft, hop, center=True))   # [T, n_bins]
    res = [analyze_mag(X[t], peak_rel_thresh, peak_count_thresh) for t in range(X.shape[0])]
    if not res:
        return {"poly_ratio": 0.0, "is_poly": 0, "n_frames": 0}
    poly = np.array([r["is_poly"] for r in res], dtype=np.float64)
    npk = np.array([r["n_peaks"] for r in res], dtype=np.float64)
    ratio = float(poly.mean())
    is_poly = int(ratio >= 0.5) if agg == "mean" else int(np.median(poly) >= 0.5)
    return {"poly_ratio": ratio, "median_peaks": float(np.median(npk)),
            "is_poly": is_poly, "n_frames": int(len(res))}


__all__ = ["analyze_mag", "analyze_audio"]
