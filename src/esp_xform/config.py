"""配置体系：dataclass 自带默认值，YAML/JSON 可选覆盖。

设计目标：即使没有安装 pyyaml，框架也能用代码内默认配置完整运行；
若安装了 pyyaml 并提供 configs/train_ddsp.yaml，则其中的键覆盖默认值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # 可选依赖
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - 环境缺失时退化
    _HAS_YAML = False


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    frame_size: int = 1536
    hop_size: int = 512
    fft_size: int = 2048


@dataclass
class FeatureConfig:
    feature_dim: int = 20
    n_mfcc: int = 13
    n_mels: int = 26
    fmin: float = 30.0
    fmax: float = 20000.0
    f0_min: float = 55.0
    f0_max: float = 1320.0
    rolloff_percent: float = 0.85
    silence_db: float = -60.0


@dataclass
class DDSPConfig:
    n_harmonics: int = 30
    n_noise_bands: int = 10
    output_dim: int = 40


@dataclass
class ModelConfig:
    embedding_dim: int = 8
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 64])
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 1])
    film_layers: List[int] = field(default_factory=lambda: [1, 2])


@dataclass
class TrainConfig:
    batch_size: int = 16
    sequence_length: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.01
    noise_loss_weight: float = 0.5
    use_stft_loss: bool = False
    stft_loss_weight: float = 0.1
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    render_every: int = 10
    grad_clip: float = 1.0


@dataclass
class PathsConfig:
    raw_wav_dir: str = "data/raw_wav"
    manifest_path: str = "data/processed/manifests/manifest.csv"
    features_dir: str = "data/processed/features"
    labels_dir: str = "data/processed/labels"
    splits_dir: str = "data/processed/splits"
    output_dir: str = "outputs"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    ddsp: DDSPConfig = field(default_factory=DDSPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ---- 派生量 ----
    @property
    def input_dim(self) -> int:
        """网络输入维度 = 吉他特征维度 + 乐器嵌入维度。"""
        return self.feature.feature_dim + self.model.embedding_dim

    @property
    def output_dim(self) -> int:
        return self.ddsp.n_harmonics + self.ddsp.n_noise_bands

    def validate(self) -> None:
        assert self.ddsp.output_dim == self.output_dim, (
            f"output_dim 不一致: ddsp.output_dim={self.ddsp.output_dim} "
            f"!= n_harmonics+n_noise_bands={self.output_dim}"
        )
        assert self.audio.fft_size >= self.audio.frame_size, "fft_size 应 >= frame_size"
        assert self.audio.hop_size > 0 and self.audio.frame_size > 0
        assert len(self.model.channels) == len(self.model.dilations), (
            "channels 与 dilations 数量必须一致"
        )
        for li in self.model.film_layers:
            assert 0 <= li < len(self.model.channels), f"film_layers 索引越界: {li}"

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


# --------------------------------------------------------------------------- #
# 辅助：dataclass <-> dict 递归转换与合并
# --------------------------------------------------------------------------- #
def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _update_dataclass(obj: Any, data: Optional[Dict[str, Any]]) -> None:
    """用 dict 递归覆盖 dataclass 实例的字段（仅覆盖已知字段）。"""
    if not data:
        return
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _read_structured_file(path: Path) -> Optional[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if _HAS_YAML:
        return yaml.safe_load(text)
    # 没有 yaml 且不是 json：放弃解析，使用默认配置
    print(
        f"[config] 警告：未安装 pyyaml，无法解析 {path.name}，将使用代码内默认配置。"
        " 如需自定义，请 `pip install pyyaml` 或改用 .json 配置。"
    )
    return None


def load_config(path: Optional[str | Path] = None) -> Config:
    """加载配置：先用默认值，再用文件覆盖。path 为空则返回纯默认配置。"""
    cfg = Config()
    if path is not None:
        p = Path(path)
        if p.exists():
            data = _read_structured_file(p)
            _update_dataclass(cfg, data)
        else:
            print(f"[config] 警告：配置文件不存在 {p}，使用默认配置。")
    cfg.validate()
    return cfg


def load_instruments(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """加载乐器映射，返回 {'instruments': {...}, 'target_instruments': [...]}。"""
    default = {
        "instruments": {"guitar": 0, "bass": 1, "piano": 2, "violin": 3},
        "target_instruments": ["bass", "piano", "violin"],
    }
    if path is None:
        return default
    p = Path(path)
    if not p.exists():
        print(f"[config] 警告：乐器配置不存在 {p}，使用默认乐器映射。")
        return default
    data = _read_structured_file(p)
    if not data:
        return default
    default.update({k: v for k, v in data.items() if v is not None})
    return default


__all__ = [
    "AudioConfig",
    "FeatureConfig",
    "DDSPConfig",
    "ModelConfig",
    "TrainConfig",
    "PathsConfig",
    "Config",
    "load_config",
    "load_instruments",
]
