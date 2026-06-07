"""谱掩码离线推理 / 渲染（对应 §5；供听感评估与 C 端数值对拍）。

加载 checkpoint → 吉他音频 STFT → MaskNet 三头 → numpy 重建链 → 音频。
含乐器切换（逐帧 schedule，等价固件"换嵌入下一帧生效"）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .audio import apply_gain_db, build_all_matrices, pitch_shift_semitones, stft
from .config import MaskConfig, _update  # type: ignore
from .model import MaskNet
from .reconstruct import reconstruct_np


def _cfg_from_dict(d: Dict) -> MaskConfig:
    cfg = MaskConfig()
    _update(cfg, d)
    cfg.validate()
    return cfg


def load_mask_model(ckpt_path: str | Path, device: str = "cpu"):
    """返回 (model, cfg, mats, mel_mean, mel_std, instruments)。"""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(state["config"])
    model = MaskNet(cfg, state["num_instruments"]).to(device).eval()
    model.load_state_dict(state["model_state"])
    mats = build_all_matrices(cfg)
    mel_mean = state.get("mel_mean")
    mel_std = state.get("mel_std")
    instruments = state.get("instruments", {})
    return model, cfg, mats, mel_mean, mel_std, instruments


def _analyze(cfg: MaskConfig, mats, mel_mean, mel_std, audio: np.ndarray):
    """吉他音频 -> (Xlin[n_bins,T], phase[n_bins,T], x[1,M,T] 标准化logmel)。"""
    s = cfg.stft
    Xc = stft(audio, s.n_fft, s.hop, center=s.center)           # [T, n_bins]
    Xlin = np.abs(Xc); phase = np.angle(Xc)
    mel_basis = mats["mel_basis"].astype(np.float64)
    Xmel = Xlin @ mel_basis.T                                   # [T, M]
    logmel = np.log(Xmel + s.log_eps)
    if mel_mean is not None and mel_std is not None:
        logmel = (logmel - np.asarray(mel_mean)) / (np.asarray(mel_std) + 1e-8)
    x = torch.tensor(logmel.T[None].astype(np.float32))        # [1, M, T]
    return Xlin.T.copy(), phase.T.copy(), x                    # [n_bins,T], [n_bins,T], [1,M,T]


@torch.no_grad()
def predict_heads(model, x: torch.Tensor, inst_id: int, device: str = "cpu"):
    """返回 (mask[M,T], dphi[P,T], noise[B,T]) numpy。"""
    inst = torch.tensor([inst_id], dtype=torch.long, device=device)
    mask, dphi, noise = model(x.to(device), inst)
    return (mask[0].cpu().numpy(), dphi[0].cpu().numpy(), noise[0].cpu().numpy())


def render_instrument(model, cfg, mats, mel_mean, mel_std, audio: np.ndarray,
                      inst_id: int, device: str = "cpu", add_noise: bool = True,
                      pitch_semitones: float = 0.0, gain_db: float = 0.0,
                      clip_mode: str = "limit") -> np.ndarray:
    """整段吉他 -> 单一目标乐器音频。

    pitch_semitones: 运行时移频/变调（半音），把转换结果整体移到目标乐器音区
                     （吉他/贝斯/尤克里里空弦音区不同）。0=不变调，建议 ±12 内。
    gain_db: 运行时输出增益（dB），作为"驱动量"。
    clip_mode: 末级削波模式 limit(干净限幅) / soft(软饱和加谐波) / hard(硬削波)。
    """
    Xlin, phase, x = _analyze(cfg, mats, mel_mean, mel_std, audio)
    mask, dphi, noise = predict_heads(model, x, inst_id, device)
    s = cfg.stft
    y = reconstruct_np(Xlin, phase, mask, dphi, noise,
                       mats["mel_inv"], mats["phase_inv"], mats["noise_fb"],
                       s.n_fft, s.hop, center=s.center, add_noise=add_noise)
    if abs(float(pitch_semitones)) > 1e-6:
        y = pitch_shift_semitones(y, float(pitch_semitones), s.n_fft, s.hop, s.sample_rate)
    if abs(float(gain_db)) > 1e-6 or clip_mode != "limit":
        y = apply_gain_db(y, float(gain_db), clip_mode)
    return y


def build_switch_schedule(T: int, instrument_ids: List[int], hop: int, sr: int,
                          switch_seconds: float = 2.0) -> np.ndarray:
    fps = max(1, int(switch_seconds * sr / hop))
    return np.array([instrument_ids[(i // fps) % len(instrument_ids)] for i in range(T)],
                    dtype=np.int64)


def render_switch(model, cfg, mats, mel_mean, mel_std, audio: np.ndarray,
                  instrument_ids: List[int], device: str = "cpu",
                  switch_seconds: float = 2.0, add_noise: bool = True,
                  pitch_semitones: float = 0.0, gain_db: float = 0.0,
                  clip_mode: str = "limit"
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """流式乐器切换：各乐器分别预测三头，按逐帧 schedule 取用后统一重建。

    pitch_semitones: 同 render_instrument，对整段输出统一变调。
    gain_db: 运行时输出增益（dB）。clip_mode: 末级削波模式 limit/soft/hard。
    """
    Xlin, phase, x = _analyze(cfg, mats, mel_mean, mel_std, audio)
    T = x.shape[-1]
    s = cfg.stft
    sched = build_switch_schedule(T, instrument_ids, s.hop, s.sample_rate, switch_seconds)
    heads = {iid: predict_heads(model, x, iid, device) for iid in set(instrument_ids)}

    M, P, B = mats["mel_inv"].shape[1], mats["phase_inv"].shape[1], mats["noise_fb"].shape[0]
    mask = np.empty((M, T), np.float32); dphi = np.empty((P, T), np.float32); noise = np.empty((B, T), np.float32)
    for t in range(T):
        mk, dp, ns = heads[int(sched[t])]
        mask[:, t] = mk[:, t]; dphi[:, t] = dp[:, t]; noise[:, t] = ns[:, t]
    y = reconstruct_np(Xlin, phase, mask, dphi, noise,
                       mats["mel_inv"], mats["phase_inv"], mats["noise_fb"],
                       s.n_fft, s.hop, center=s.center, add_noise=add_noise)
    if abs(float(pitch_semitones)) > 1e-6:
        y = pitch_shift_semitones(y, float(pitch_semitones), s.n_fft, s.hop, s.sample_rate)
    if abs(float(gain_db)) > 1e-6 or clip_mode != "limit":
        y = apply_gain_db(y, float(gain_db), clip_mode)
    return y, sched


__all__ = ["load_mask_model", "render_instrument", "render_switch",
           "build_switch_schedule", "predict_heads"]
