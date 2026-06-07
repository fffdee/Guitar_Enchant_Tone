"""谱掩码方案的 STFT / ISTFT / OLA 与各展开矩阵（纯 numpy，对应附录 E）。

这是数值"源头"：C 端 frontend/synth 必须逐一镜像本文件的算法与矩阵，
以保证 PC 训练与 MCU 部署数值一致。

约定：
  - 加权 OLA：分析/合成都加同一窗，重建时除以 Σ(window^2)，对任意 COLA 窗都精确重建；
  - center=True：两端各零填充 n_fft//2，使第 t 帧中心对齐样点 t*hop；
  - 复数谱以 numpy rfft 约定（长度 n_bins=n_fft//2+1）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..audio.resample import resample_to
from ..features.mfcc import mel_filterbank  # 复用 HTK 三角梅尔 (与 C 端一致)

_EPS = 1e-8


def hann_window(n: int) -> np.ndarray:
    """周期 Hann 窗 (0.5 - 0.5 cos(2πk/n))，hop=n/4 时满足 COLA。"""
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float64)


def stft(x: np.ndarray, n_fft: int, hop: int, center: bool = True,
         window: Optional[np.ndarray] = None) -> np.ndarray:
    """返回复数谱 [T, n_bins]。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if window is None:
        window = hann_window(n_fft)
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect" if x.size >= n_fft else "constant")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n_frames = 1 + (x.size - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.complex128)
    for t in range(n_frames):
        seg = x[t * hop: t * hop + n_fft] * window
        out[t] = np.fft.rfft(seg, n=n_fft)
    return out


def istft(X: np.ndarray, n_fft: int, hop: int, center: bool = True,
          window: Optional[np.ndarray] = None, length: Optional[int] = None) -> np.ndarray:
    """复数谱 [T, n_bins] -> 时域，加权 OLA + Σw^2 归一。"""
    X = np.asarray(X, dtype=np.complex128)
    if window is None:
        window = hann_window(n_fft)
    T = X.shape[0]
    out_len = (T - 1) * hop + n_fft
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    w2 = window * window
    for t in range(T):
        frame = np.fft.irfft(X[t], n=n_fft) * window
        s = t * hop
        y[s: s + n_fft] += frame
        wsum[s: s + n_fft] += w2
    y = y / np.maximum(wsum, _EPS)
    if center:
        y = y[n_fft // 2: len(y) - n_fft // 2]
    if length is not None:
        if len(y) >= length:
            y = y[:length]
        else:
            y = np.pad(y, (0, length - len(y)))
    return y.astype(np.float32)


def stft_mag_phase(x: np.ndarray, n_fft: int, hop: int, center: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (幅度 [T,n_bins], 相位 [T,n_bins])。"""
    X = stft(x, n_fft, hop, center)
    return np.abs(X).astype(np.float32), np.angle(X).astype(np.float32)


def _a_weight_lin(freqs: np.ndarray) -> np.ndarray:
    """A 计权(线性幅度)曲线，近似人耳等响度：强烈抑制极低频/极高频。

    仅用于"感知响度比值"，故无需归一到 0dB@1kHz（常数在比值中抵消）。
    """
    f = np.asarray(freqs, dtype=np.float64)
    f2 = f * f
    num = (12194.0 ** 2) * (f2 * f2)
    den = ((f2 + 20.6 ** 2)
           * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
           * (f2 + 12194.0 ** 2)) + 1e-30
    return num / den


def _perceptual_rms(y: np.ndarray, sr: int) -> float:
    """A 计权感知响度（频域 Parseval）。低频被抑制，更贴近"听起来多响"。"""
    n = y.size
    if n < 2:
        return 0.0
    X = np.abs(np.fft.rfft(y))
    w = _a_weight_lin(np.fft.rfftfreq(n, 1.0 / sr))
    Xw = X * w
    return float(np.sqrt(np.sum(Xw * Xw)) / n)


def apply_gain_db(y: np.ndarray, gain_db: float = 0.0,
                  clip_mode: str = "limit", peak_limit: float = 0.999) -> np.ndarray:
    """对输出施加 dB 增益 + 末级削波模式。增益当"驱动量"，模式决定如何处理过冲。

    gain_db: 增益(dB)，>0 提升 / <0 衰减 / 0 不变。
    clip_mode:
      - "limit"(默认): 干净限幅——峰值超 peak_limit 则整体回缩，不失真、不过载。
      - "soft" : 软饱和 tanh(y)，超过 ±1 平滑压缩并产生谐波(让超低频在小喇叭上"听得见"、更厚)。
      - "hard" : 硬削波 clip 到 [-1,1]，最强过载/失真。
    peak_limit: 仅 "limit" 模式生效。
    """
    g = 10.0 ** (float(gain_db) / 20.0)
    out = np.asarray(y, dtype=np.float64) * g
    if clip_mode == "soft":
        out = np.tanh(out)
    elif clip_mode == "hard":
        out = np.clip(out, -1.0, 1.0)
    else:  # limit
        if peak_limit and peak_limit > 0:
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > peak_limit:
                out *= peak_limit / peak
    return out.astype(np.float32)


def pitch_shift_semitones(y: np.ndarray, n_semitones: float,
                          n_fft: int = 1024, hop: int = 256,
                          sr: int = 48000) -> np.ndarray:
    """相位声码器变调（保持时长）：把整段音频升/降 n_semitones 个半音。

    用途：谱掩码只改音色不改音高；本函数作为重建链后的"移频/变调"控制，
    把转换结果整体移到目标乐器的自然音区（吉他/贝斯/尤克里里空弦音区不同）。

    做法：相位声码器按比例 r=2^(n/12) 做时间伸缩（合成 hop=hop·r，相位逐帧累加
    真实频率估计），再按长度重采样回原长 → 音高×r、时长不变。
    建议范围约 ±12 半音（一个八度内）以保证音质。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if abs(n_semitones) < 1e-6 or y.size < n_fft:
        return y.astype(np.float32)
    ratio = 2.0 ** (float(n_semitones) / 12.0)

    win = hann_window(n_fft)
    D = stft(y, n_fft, hop, center=True, window=win)        # [T, bins]
    T, bins = D.shape
    mag = np.abs(D)
    ph = np.angle(D)
    omega = 2.0 * np.pi * hop * np.arange(bins) / n_fft      # 每个分析 hop 的期望相位推进
    Hs = max(1, int(round(hop * ratio)))                    # 合成 hop

    out = np.empty_like(D)
    acc = ph[0].copy()
    out[0] = mag[0] * np.exp(1j * acc)
    scale = Hs / float(hop)
    for t in range(1, T):
        dphi = ph[t] - ph[t - 1] - omega
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))   # 相位卷绕到 [-pi,pi]
        acc = acc + scale * (omega + dphi)                  # 按合成 hop 推进
        out[t] = mag[t] * np.exp(1j * acc)

    stretched = istft(out, n_fft, Hs, center=True, window=win)      # 时长 ≈ 原长×ratio
    if stretched.size < 1:
        return y.astype(np.float32)
    shifted = resample_to(stretched, stretched.size, y.size)        # 重采样回原长 → 音高×ratio
    if shifted.size < y.size:                                       # 长度对齐保护
        shifted = np.pad(shifted, (0, y.size - shifted.size))
    shifted = shifted[: y.size].astype(np.float64)

    # 保持"感知响度"：变调本不应改变响度。降调把能量搬到低频，人耳/音箱对低频不敏感，
    # 故仅按物理 RMS 归一仍会"听起来变小"。改用 A 计权感知响度匹配，用余量把降调结果提上来；
    # 升调则相应压低，避免发尖。最后做削波保护。
    in_l = _perceptual_rms(y, sr)
    out_l = _perceptual_rms(shifted, sr)
    if in_l > 1e-9 and out_l > 1e-9:
        gain = in_l / out_l
        gain = float(np.clip(gain, 0.125, 8.0))     # 安全上下限，避免极端放大噪声
        shifted *= gain
    peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
    if peak > 0.999:                                  # 削波保护（优先不失真）
        shifted *= 0.999 / peak
    return shifted.astype(np.float32)


# --------------------------------------------------------------------------- #
# 矩阵构建（离线一次，导出给 C 端预存）
# --------------------------------------------------------------------------- #
def build_mel_basis(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """梅尔滤波器组 Mel[n_mels, n_bins]。"""
    return mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).astype(np.float32)


def build_mel_inv(mel_basis: np.ndarray) -> np.ndarray:
    """梅尔→线性 增益展开矩阵 MelInv[n_bins, n_mels]，按 bin 归一避免叠加放大。

    gain_lin[r] = Σ_m MelInv[r,m] · mask_mel[m] = 各覆盖该 bin 的梅尔增益加权平均 ∈ [0,Gmax]。
    """
    mel_basis = np.asarray(mel_basis, dtype=np.float64)         # [M, n_bins]
    bin_sum = mel_basis.sum(axis=0)                             # [n_bins]
    mel_inv = mel_basis.T / (bin_sum[:, None] + _EPS)           # [n_bins, M]
    return mel_inv.astype(np.float32)


def build_phase_inv(n_bins: int, phase_bands: int) -> np.ndarray:
    """相位残差 低分辨率(P) -> 线性(n_bins) 的线性插值矩阵 PhaseInv[n_bins, P]。

    band 中心在 [0, n_bins-1] 上等距；每个 bin 由相邻两 band 线性插值。
    """
    centers = np.linspace(0.0, n_bins - 1, phase_bands)
    M = np.zeros((n_bins, phase_bands), dtype=np.float64)
    for r in range(n_bins):
        # 定位 r 落在哪两个 band 中心之间
        if r <= centers[0]:
            M[r, 0] = 1.0
            continue
        if r >= centers[-1]:
            M[r, -1] = 1.0
            continue
        j = np.searchsorted(centers, r) - 1
        j = max(0, min(j, phase_bands - 2))
        lo, hi = centers[j], centers[j + 1]
        frac = (r - lo) / max(hi - lo, _EPS)
        M[r, j] = 1.0 - frac
        M[r, j + 1] = frac
    return M.astype(np.float32)


def build_noise_fb(sr: int, n_fft: int, noise_bands: int, fmin: float, fmax: float) -> np.ndarray:
    """噪声带滤波器组 NoiseFB[noise_bands, n_bins]（粗梅尔三角，用于把带增益展开成谱形）。"""
    return mel_filterbank(sr, n_fft, n_mels=noise_bands, fmin=fmin, fmax=fmax).astype(np.float32)


def build_all_matrices(cfg) -> dict:
    """按 MaskConfig 构建全部展开矩阵，返回 dict。"""
    s = cfg.stft
    mel_basis = build_mel_basis(s.sample_rate, s.n_fft, s.n_mels, s.fmin, s.fmax)
    mel_inv = build_mel_inv(mel_basis)
    phase_inv = build_phase_inv(s.n_bins, cfg.model.phase_bands)
    noise_fb = build_noise_fb(s.sample_rate, s.n_fft, cfg.model.noise_bands, s.fmin, s.fmax)
    return {
        "mel_basis": mel_basis,      # [M, n_bins]
        "mel_inv": mel_inv,          # [n_bins, M]
        "phase_inv": phase_inv,      # [n_bins, P]
        "noise_fb": noise_fb,        # [B, n_bins]
    }


def log_mel_from_mag(mag: np.ndarray, mel_basis: np.ndarray, log_eps: float = 1e-5) -> np.ndarray:
    """幅度谱 [T,n_bins] -> 对数梅尔 [T,M]。"""
    mel = np.asarray(mag, dtype=np.float64) @ np.asarray(mel_basis, dtype=np.float64).T
    return np.log(mel + log_eps).astype(np.float32)


__all__ = [
    "hann_window",
    "stft",
    "istft",
    "stft_mag_phase",
    "pitch_shift_semitones",
    "apply_gain_db",
    "build_mel_basis",
    "build_mel_inv",
    "build_phase_inv",
    "build_noise_fb",
    "build_all_matrices",
    "log_mel_from_mag",
]
