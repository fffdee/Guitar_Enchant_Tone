"""推理辅助：加载模型、由特征预测 DDSP 参数、合成音频、流式切换乐器渲染。

供训练期 demo 渲染回调与独立渲染脚本共用，保证训练/推理路径一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .ddsp.bands import make_band_edges
from .ddsp.synthesizer import ddsp_synth_np
from .features.guitar_features import GuitarFeatureExtractor
from .models import ConditionalDDSPNet


def _cfg_from_dict(d: Dict) -> Config:
    from .config import load_config
    cfg = load_config(None)
    from .config import _update_dataclass  # type: ignore
    _update_dataclass(cfg, d)
    cfg.validate()
    return cfg


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    """加载 checkpoint，返回 (model, cfg, feature_mean, feature_std, instruments, band_edges)。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(ckpt["config"])
    num_inst = ckpt["num_instruments"]
    model = ConditionalDDSPNet(cfg, num_inst).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt.get("feature_mean")
    std = ckpt.get("feature_std")
    instruments = ckpt.get("instruments", {})
    band_edges = make_band_edges(cfg.audio.sample_rate, cfg.audio.fft_size, cfg.ddsp.n_noise_bands)
    return model, cfg, mean, std, instruments, band_edges


@torch.no_grad()
def predict_params(
    model,
    features_raw: np.ndarray,
    instrument_id: int,
    feature_mean: Optional[np.ndarray],
    feature_std: Optional[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """由原始特征预测 DDSP 参数 (T,40)，已标准化输入并 clamp 非负输出。"""
    feats = features_raw.astype(np.float32)
    if feature_mean is not None and feature_std is not None:
        feats = (feats - feature_mean) / (feature_std + 1e-7)
    x = torch.from_numpy(feats).unsqueeze(0).to(device)            # (1,T,20)
    inst = torch.tensor([instrument_id], dtype=torch.long, device=device)
    params = model(x, inst)
    params = ConditionalDDSPNet.to_synth_params(params)
    return params.squeeze(0).cpu().numpy().astype(np.float32)


def synth_from_params(
    params: np.ndarray,
    f0_hz: np.ndarray,
    cfg: Config,
    band_edges: np.ndarray,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """由参数 (T,40) + 逐帧 f0 合成音频（numpy）。"""
    nh = cfg.ddsp.n_harmonics
    harmonic, noise = params[:, :nh], params[:, nh:]
    n_samples = params.shape[0] * cfg.audio.hop_size
    return ddsp_synth_np(
        f0_hz, harmonic, noise,
        cfg.audio.sample_rate, cfg.audio.hop_size, cfg.audio.fft_size,
        band_edges, n_samples=n_samples, seed=seed,
    )


def render_instrument(
    model, source_audio: np.ndarray, instrument_id: int, cfg: Config,
    feature_mean, feature_std, band_edges, device: str = "cpu",
) -> np.ndarray:
    """对整段源吉他音频以单一目标乐器渲染。"""
    ext = GuitarFeatureExtractor(cfg)
    feat = ext.extract(source_audio)
    if feat["n_frames"] == 0:
        return np.zeros(0, dtype=np.float32)
    params = predict_params(model, feat["features"], instrument_id, feature_mean, feature_std, device)
    return synth_from_params(params, feat["f0_hz"], cfg, band_edges)


def build_switch_schedule(n_frames: int, instrument_ids: List[int], hop_size: int,
                          sample_rate: int, switch_seconds: float = 2.0) -> np.ndarray:
    """生成逐帧乐器 ID 序列，每 switch_seconds 轮换一个目标乐器。"""
    frames_per_seg = max(1, int(switch_seconds * sample_rate / hop_size))
    sched = np.empty(n_frames, dtype=np.int64)
    for i in range(n_frames):
        seg = (i // frames_per_seg) % len(instrument_ids)
        sched[i] = instrument_ids[seg]
    return sched


def streaming_render_switch(
    model, source_audio: np.ndarray, instrument_ids: List[int], cfg: Config,
    feature_mean, feature_std, band_edges, device: str = "cpu",
    switch_seconds: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """流式切换渲染：同一段吉他，按时间轮换目标乐器（验证零成本切换）。

    返回 (audio, schedule)。各乐器参数在整段上分别预测，再逐帧按 schedule 取用，
    等价于固件中“切换嵌入、下一帧即生效”。
    """
    ext = GuitarFeatureExtractor(cfg)
    feat = ext.extract(source_audio)
    T = int(feat["n_frames"])
    if T == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    sched = build_switch_schedule(T, instrument_ids, cfg.audio.hop_size,
                                  cfg.audio.sample_rate, switch_seconds)
    params_by_inst = {
        iid: predict_params(model, feat["features"], iid, feature_mean, feature_std, device)
        for iid in set(instrument_ids)
    }
    out_params = np.empty((T, cfg.output_dim), dtype=np.float32)
    for i in range(T):
        out_params[i] = params_by_inst[int(sched[i])][i]
    audio = synth_from_params(out_params, feat["f0_hz"], cfg, band_edges)
    return audio, sched


__all__ = [
    "load_model",
    "predict_params",
    "synth_from_params",
    "render_instrument",
    "build_switch_schedule",
    "streaming_render_switch",
]
