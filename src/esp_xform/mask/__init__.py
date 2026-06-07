"""频域谱掩码方案（复音实时音色转换主线，对应 IMPLEMENTATION.md）。

与原 DDSP 方案并存：谱掩码为主线（原生复音），DDSP 退为单音深变换模式(§8)。
"""

from .config import MaskConfig, load_mask_config

__all__ = ["MaskConfig", "load_mask_config"]
