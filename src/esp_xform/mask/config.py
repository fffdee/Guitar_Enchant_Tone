"""频域谱掩码方案配置（对应 IMPLEMENTATION.md 附录 B，唯一参数来源）。

与原 DDSP 配置（esp_xform.config）并存：谱掩码为复音主线，DDSP 退为单音模式(§8)。
所有默认值取自附录 B；可用 YAML/JSON 覆盖（缺 pyyaml 时回退默认）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # 可选依赖
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


@dataclass
class STFTConfig:
    sample_rate: int = 48000
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 96
    fmin: float = 40.0
    fmax: float = 16000.0
    center: bool = True
    log_eps: float = 1e-5      # log(mel + eps)
    mask_eps: float = 1e-4     # 掩码分母下限

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop


@dataclass
class MaskModelConfig:
    hidden: int = 128
    kernel: int = 3
    dilation2: int = 2          # L2 膨胀
    emb_dim: int = 16           # 乐器嵌入
    n_artic: int = 0            # 演奏法类别数 (0=不启用)
    artic_dim: int = 4
    gmax: float = 4.0           # 幅度掩码上限
    phase_bands: int = 64       # 相位残差头输出 P
    dphi_max: float = math.pi / 2.0
    noise_bands: int = 16       # 噪声带头输出 B
    causal: bool = False        # 实时流式可设 True (左侧 padding)


@dataclass
class MaskTrainConfig:
    win_frames: int = 128       # 训练窗帧数 T (≈0.68s)
    win_stride: int = 64        # 切窗步长
    batch_size: int = 32
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # 损失权重 (附录 B.3)
    w_mask: float = 1.0
    w_logmel: float = 0.5
    w_mrstft: float = 0.1
    w_cplx: float = 0.1         # 启用相位残差头
    w_adv: float = 0.01         # 后期微调
    use_phase: bool = True
    use_adv: bool = False
    augment: bool = False       # §4.4 输入域增强 (仅训练集)
    # 多分辨率 STFT 配置 (n_fft, hop)
    mrstft_ffts: List[int] = field(default_factory=lambda: [2048, 1024, 512])
    mrstft_hops: List[int] = field(default_factory=lambda: [512, 256, 128])
    render_every: int = 10


@dataclass
class MaskPathsConfig:
    raw_dir: str = "dataset/raw"
    proc_dir: str = "dataset/proc"
    output_dir: str = "outputs/mask"


@dataclass
class MaskConfig:
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: MaskModelConfig = field(default_factory=MaskModelConfig)
    train: MaskTrainConfig = field(default_factory=MaskTrainConfig)
    paths: MaskPathsConfig = field(default_factory=MaskPathsConfig)

    @property
    def cond_dim(self) -> int:
        d = self.model.emb_dim
        if self.model.n_artic > 0:
            d += self.model.artic_dim
        return d

    def validate(self) -> None:
        s = self.stft
        assert s.n_fft > 0 and (s.n_fft & (s.n_fft - 1)) == 0, "n_fft 必须为 2 的幂 (MCU FFT 约束)"
        assert s.hop > 0 and s.n_fft % s.hop == 0, "n_fft 应为 hop 的整数倍 (COLA)"
        assert 0 < s.fmin < s.fmax <= s.sample_rate / 2.0
        assert self.model.gmax > 0 and self.model.dphi_max > 0
        assert self.model.phase_bands > 0 and self.model.noise_bands > 0
        assert len(self.train.mrstft_ffts) == len(self.train.mrstft_hops)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


# --------------------------------------------------------------------------- #
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _update(obj: Any, data: Optional[Dict[str, Any]]) -> None:
    if not data:
        return
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update(current, value)
        else:
            setattr(obj, key, value)


def load_mask_config(path: Optional[str | Path] = None) -> MaskConfig:
    cfg = MaskConfig()
    if path is not None:
        p = Path(path)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if p.suffix.lower() == ".json":
                _update(cfg, json.loads(text))
            elif _HAS_YAML:
                _update(cfg, yaml.safe_load(text))
            else:
                print(f"[mask.config] 未安装 pyyaml，无法解析 {p.name}，使用默认配置。")
        else:
            print(f"[mask.config] 配置不存在 {p}，使用默认配置。")
    cfg.validate()
    return cfg


__all__ = [
    "STFTConfig",
    "MaskModelConfig",
    "MaskTrainConfig",
    "MaskPathsConfig",
    "MaskConfig",
    "load_mask_config",
]
