"""谱掩码方案：数据预处理 + 训练数据集（对应 §2 / 附录 E）。

预处理策略：每个 (片段, 目标乐器) 对，对齐后存整段波形 (float16) + 窗口索引；
训练时按窗切片即时算 STFT→梅尔→掩码标签（省存储、保证与重建同一套 STFT）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..audio.io import load_wav
from .audio import build_all_matrices, hann_window, stft

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 对齐（同一 MIDI 双渲染：常数偏移即可，§2.3 不用 DTW）
# --------------------------------------------------------------------------- #
def _rms_env(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n <= 0:
        return np.zeros(1)
    x = x[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(x * x, axis=1) + _EPS)


def _estimate_offset_samples(g: np.ndarray, t: np.ndarray, sr: int, hop: int, max_ms: float = 50.0) -> int:
    eg, et = _rms_env(g, hop), _rms_env(t, hop)
    eg = eg - eg.mean()
    et = et - et.mean()
    max_lag = max(1, int(max_ms / 1000.0 * sr / hop))
    n = min(len(eg), len(et))
    eg, et = eg[:n], et[:n]
    best_lag, best_val = 0, -1e30
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = eg[lag:], et[: n - lag]
        else:
            a, b = eg[: n + lag], et[-lag:]
        if len(a) < 4:
            continue
        v = float(np.dot(a, b))
        if v > best_val:
            best_val, best_lag = v, lag
    return best_lag * hop


def _align_pair(g: np.ndarray, t: np.ndarray, sr: int, hop: int) -> tuple:
    off = _estimate_offset_samples(g, t, sr, hop)
    if off > 0:        # 目标比吉他晚 off 个样点 -> 目标左移
        t = t[off:]
    elif off < 0:
        g = g[-off:]
    n = min(len(g), len(t))
    return g[:n], t[:n]


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def preprocess(cfg, instruments: Dict[str, int], limit: Optional[int] = None) -> Dict[str, str]:
    """遍历 raw/clip_*/, 对齐 → 存波形对 + 窗口索引 + 梅尔统计 + 展开矩阵。"""
    s = cfg.stft
    raw = Path(cfg.paths.raw_dir)
    proc = Path(cfg.paths.proc_dir)
    pairs_dir = proc / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    mats = build_all_matrices(cfg)
    for k, v in mats.items():
        np.save(proc / f"{k}.npy", v)
    mel_basis = mats["mel_basis"].astype(np.float64)  # [M, n_bins]

    T, stride, hop = cfg.train.win_frames, cfg.train.win_stride, s.hop
    win_samples = T * hop

    index: List[Dict] = []
    # 梅尔统计累计（吉他 log_mel 逐通道）
    mel_sum = np.zeros(s.n_mels, dtype=np.float64)
    mel_sqsum = np.zeros(s.n_mels, dtype=np.float64)
    mel_count = 0
    pair_id = 0

    clips = sorted([d for d in raw.glob("clip_*") if d.is_dir()])
    if limit:
        clips = clips[:limit]
    for clip in clips:
        gpath = clip / "guitar.wav"
        if not gpath.exists():
            continue
        g, _ = load_wav(gpath, expected_sr=s.sample_rate, mono=True)
        for tpath in sorted(clip.glob("*.wav")):
            name = tpath.stem
            if name == "guitar" or name not in instruments:
                continue
            inst_id = int(instruments[name])
            t, _ = load_wav(tpath, expected_sr=s.sample_rate, mono=True)
            ga, ta = _align_pair(g, t, s.sample_rate, hop)
            if len(ga) < win_samples:
                continue

            # 存波形对 (float16)
            fname = f"pair_{pair_id:05d}.npz"
            np.savez(pairs_dir / fname,
                     g=ga.astype(np.float16), t=ta.astype(np.float16), inst_id=inst_id)

            # 统计 + 切窗索引
            Xc = stft(ga, s.n_fft, s.hop, center=s.center)         # [Tc, n_bins]
            Xmag = np.abs(Xc)
            Xmel = Xmag @ mel_basis.T                              # [Tc, M]
            logmel = np.log(Xmel + s.log_eps)
            mel_sum += logmel.sum(axis=0)
            mel_sqsum += (logmel * logmel).sum(axis=0)
            mel_count += logmel.shape[0]

            n_frames = len(ga) // hop
            for start in range(0, max(1, n_frames - T + 1), stride):
                index.append({"file": fname, "start": int(start)})
            pair_id += 1

    mel_mean = mel_sum / max(mel_count, 1)
    mel_std = np.sqrt(np.maximum(mel_sqsum / max(mel_count, 1) - mel_mean**2, 1e-8))
    np.savez(proc / "stats.npz",
             mel_mean=mel_mean.astype(np.float32), mel_std=mel_std.astype(np.float32))
    (proc / "windows.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"[mask.preprocess] pairs={pair_id} windows={len(index)} mel_frames={mel_count}")
    return {"proc_dir": str(proc), "n_pairs": str(pair_id), "n_windows": str(len(index))}


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class MaskWindowDataset(Dataset):
    """按窗返回：x(标准化logmel)、mask_lab、Xmel、Ymel、Xlin、phase、target_wav、inst_id。

    augment: 可选 AugmentConfig，仅对输入吉他做域增强（§4.4），掎码标签因即时重算而自洽。
    """

    def __init__(self, cfg, proc_dir: Optional[str] = None, augment=None) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("MaskWindowDataset 需要 torch")
        self.cfg = cfg
        self.s = cfg.stft
        proc = Path(proc_dir or cfg.paths.proc_dir)
        self.pairs_dir = proc / "pairs"
        self.index = json.loads((proc / "windows.json").read_text(encoding="utf-8"))
        self.mel_basis = np.load(proc / "mel_basis.npy").astype(np.float64)   # [M, n_bins]
        st = np.load(proc / "stats.npz")
        self.mel_mean = st["mel_mean"].astype(np.float64)
        self.mel_std = st["mel_std"].astype(np.float64)
        self.T = cfg.train.win_frames
        self.hop = self.s.hop
        self.win_samples = self.T * self.hop
        self.augment = augment
        self._rng = np.random.default_rng(cfg.train.seed)
        self._cache_file = None
        self._cache = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_pair(self, fname: str):
        if fname != self._cache_file:
            self._cache = np.load(self.pairs_dir / fname)
            self._cache_file = fname
        return self._cache

    def __getitem__(self, i: int):
        rec = self.index[i]
        d = self._load_pair(rec["file"])
        start = rec["start"]
        s0 = start * self.hop
        gw = np.asarray(d["g"][s0: s0 + self.win_samples], dtype=np.float64)
        tw = np.asarray(d["t"][s0: s0 + self.win_samples], dtype=np.float64)
        if len(gw) < self.win_samples:
            gw = np.pad(gw, (0, self.win_samples - len(gw)))
        if len(tw) < self.win_samples:
            tw = np.pad(tw, (0, self.win_samples - len(tw)))
        inst_id = int(d["inst_id"])

        if self.augment is not None:
            from .augment import augment_input
            gw = augment_input(gw, self.s.sample_rate, self.augment, self._rng).astype(np.float64)

        T = self.T
        Xc = stft(gw, self.s.n_fft, self.s.hop, center=self.s.center)[:T]   # [T,n_bins]
        Tc = stft(tw, self.s.n_fft, self.s.hop, center=self.s.center)[:T]
        Xlin = np.abs(Xc); phase = np.angle(Xc)
        Ylin = np.abs(Tc); Yphase = np.angle(Tc)
        Xmel = Xlin @ self.mel_basis.T                                       # [T,M]
        Ymel = Ylin @ self.mel_basis.T
        logmel = np.log(Xmel + self.s.log_eps)
        x = (logmel - self.mel_mean) / self.mel_std
        mask_lab = np.clip(Ymel / (Xmel + self.s.mask_eps), 0.0, self.cfg.model.gmax)

        # 转 [C, T]
        def tt(a):
            return torch.tensor(a.T.copy(), dtype=torch.float32)
        return {
            "x": tt(x),                 # [M,T]
            "mask_lab": tt(mask_lab),   # [M,T]
            "Xmel": tt(Xmel),           # [M,T]
            "Ymel": tt(Ymel),           # [M,T]
            "Xlin": tt(Xlin),           # [n_bins,T]
            "phase": tt(phase),         # [n_bins,T]
            "Ymag": tt(Ylin),           # [n_bins,T]
            "Yphase": tt(Yphase),       # [n_bins,T]
            "target_wav": torch.tensor(tw, dtype=torch.float32),  # [win_samples]
            "inst_id": torch.tensor(inst_id, dtype=torch.long),
        }


class MaskWaveDataset(Dataset):
    """只返回原始波形的数据集（用于GPU加速STFT）。

    返回：吉他波形、目标波形、乐器ID。STFT计算在训练循环中使用PyTorch GPU完成。
    """

    def __init__(self, cfg, proc_dir: Optional[str] = None, augment=None) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("MaskWaveDataset 需要 torch")
        self.cfg = cfg
        self.s = cfg.stft
        proc = Path(proc_dir or cfg.paths.proc_dir)
        self.pairs_dir = proc / "pairs"
        self.index = json.loads((proc / "windows.json").read_text(encoding="utf-8"))
        st = np.load(proc / "stats.npz")
        self.mel_mean = torch.tensor(st["mel_mean"], dtype=torch.float32)
        self.mel_std = torch.tensor(st["mel_std"], dtype=torch.float32)
        self.mel_basis = torch.tensor(np.load(proc / "mel_basis.npy"), dtype=torch.float32)
        self.T = cfg.train.win_frames
        self.hop = self.s.hop
        self.win_samples = self.T * self.hop
        self.augment = augment
        self._rng = np.random.default_rng(cfg.train.seed)
        self._cache_file = None
        self._cache = None
        self._window = torch.tensor(hann_window(self.s.n_fft), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.index)

    def _load_pair(self, fname: str):
        if fname != self._cache_file:
            self._cache = np.load(self.pairs_dir / fname)
            self._cache_file = fname
        return self._cache

    def __getitem__(self, i: int):
        rec = self.index[i]
        d = self._load_pair(rec["file"])
        start = rec["start"]
        s0 = start * self.hop
        gw = np.asarray(d["g"][s0: s0 + self.win_samples], dtype=np.float32)
        tw = np.asarray(d["t"][s0: s0 + self.win_samples], dtype=np.float32)
        if len(gw) < self.win_samples:
            gw = np.pad(gw, (0, self.win_samples - len(gw)))
        if len(tw) < self.win_samples:
            tw = np.pad(tw, (0, self.win_samples - len(tw)))
        inst_id = int(d["inst_id"])

        if self.augment is not None:
            from .augment import augment_input
            gw = augment_input(gw, self.s.sample_rate, self.augment, self._rng).astype(np.float32)

        return {
            "g_wav": torch.tensor(gw, dtype=torch.float32),  # [win_samples]
            "t_wav": torch.tensor(tw, dtype=torch.float32),  # [win_samples]
            "inst_id": torch.tensor(inst_id, dtype=torch.long),
        }


def collate(batch: List[Dict]):
    import torch
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


__all__ = ["preprocess", "MaskWindowDataset", "collate"]
