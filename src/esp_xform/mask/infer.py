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
from .export import collect_conv1d_int8_weights
from .model import MaskNet
from .reconstruct import reconstruct_np


def _conv1d_out_len(T: int, K: int, S: int, P: int, D: int) -> int:
    return (T + 2 * P - D * (K - 1) - 1) // S + 1


def conv1d_int8_np(x: np.ndarray, wp: dict) -> np.ndarray:
    """Conv1d INT8 前向 (MCU layer_conv1d 对齐, dilation=1).

    x: [Cin, T] f32 -> y: [Cout, To] f32
    输入 per-tensor 量化; 权重 per-output-channel; im2col + int32 MAC + 反量化 + bias.
    """
    x = np.asarray(x, dtype=np.float32)
    Cin, T = x.shape
    W_i8 = wp["W_i8"]
    Cout, Cin_K = W_i8.shape
    K, S, P, D = wp["K"], wp["stride"], wp["padding"], wp["dilation"]
    if D != 1:
        raise ValueError("conv1d_int8_np 仅支持 dilation=1 (MCU INT8 路径限制)")
    if Cin * K != Cin_K:
        raise ValueError(f"Cin*K mismatch: {Cin}*{K} vs Cin_K={Cin_K}")

    To = _conv1d_out_len(T, K, S, P, D)
    scale_in = max(float(np.max(np.abs(x))), 1e-8) / 127.0
    x_i8 = np.clip(np.round(x / scale_in), -128, 127).astype(np.int8)

    col_i8 = np.zeros((To, Cin_K), dtype=np.int32)
    for ci in range(Cin):
        xrow = x_i8[ci]
        for k in range(K):
            col_idx = ci * K + k
            base = k * D - P
            for t in range(To):
                idx = t * S + base
                if 0 <= idx < T:
                    col_i8[t, col_idx] = int(xrow[idx])

    acc = col_i8 @ W_i8.astype(np.int32).T                     # [To, Cout]
    out_scale = scale_in * wp["scale"].astype(np.float64)     # [Cout]
    y = acc.astype(np.float64) * out_scale[None, :] + wp["bias_f32"][None, :]
    return y.T.astype(np.float32)                             # [Cout, To]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _film_np(x: np.ndarray, film_mod, cond: torch.Tensor) -> np.ndarray:
    """FiLM 保持 F32 (MCU 亦不走 INT8). x:[C,T]"""
    xt = torch.from_numpy(x[None].astype(np.float32))
    with torch.no_grad():
        y = film_mod(xt, cond)
    return y[0].cpu().numpy()


def _conv_f32_np(x: np.ndarray, model: MaskNet, conv: torch.nn.Conv1d, dilation: int) -> np.ndarray:
    xt = torch.from_numpy(x[None].astype(np.float32))
    with torch.no_grad():
        y = model._conv(xt, conv, dilation)
    return y[0].cpu().numpy()


@torch.no_grad()
def predict_heads_int8(model: MaskNet, x: torch.Tensor, inst_id: int, device: str = "cpu"):
    """模拟 MCU INT8 conv 路径 (dilation=1 层 INT8, 其余 F32), 返回 mask/dphi/noise numpy."""
    x_np = x[0].cpu().numpy().astype(np.float32)
    inst = torch.tensor([inst_id], dtype=torch.long, device=device)
    cond = model.cond_vector(inst)
    wpacks = collect_conv1d_int8_weights(model)
    wi = 0

    def run_conv(xin: np.ndarray, conv: torch.nn.Conv1d) -> np.ndarray:
        nonlocal wi
        wp = wpacks[wi]
        wi += 1
        if wp["dilation"] != 1:
            return _conv_f32_np(xin, model, conv, wp["dilation"])
        return conv1d_int8_np(xin, wp)

    h = run_conv(x_np, model.c1)
    h = _film_np(_relu(h), model.f1, cond)
    h = run_conv(h, model.c2)
    h = _film_np(_relu(h), model.f2, cond)
    h = _relu(run_conv(h, model.c3))

    h_mag = run_conv(h, model.head_mag)
    mask = (1.0 / (1.0 + np.exp(-h_mag))) * float(model.gmax)

    h_phase = run_conv(h, model.head_phase)
    dphi = np.tanh(h_phase) * float(model.dphi_max)

    h_noise = run_conv(h, model.head_noise)
    noise = np.log1p(np.exp(h_noise))

    return mask, dphi, noise


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
                      clip_mode: str = "limit", use_int8: bool = False) -> np.ndarray:
    """整段吉他 -> 单一目标乐器音频。

    pitch_semitones: 运行时移频/变调（半音），把转换结果整体移到目标乐器音区
                     （吉他/贝斯/尤克里里空弦音区不同）。0=不变调，建议 ±12 内。
    gain_db: 运行时输出增益（dB），作为"驱动量"。
    clip_mode: 末级削波模式 limit(干净限幅) / soft(软饱和加谐波) / hard(硬削波)。
    use_int8: True 时用 MCU 对齐的 INT8 conv 路径 (dilation=1 层), 用于试听/对拍。
    """
    Xlin, phase, x = _analyze(cfg, mats, mel_mean, mel_std, audio)
    if use_int8:
        mask, dphi, noise = predict_heads_int8(model, x, inst_id, device)
    else:
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
           "build_switch_schedule", "predict_heads", "predict_heads_int8",
           "conv1d_int8_np"]
