"""从 checkpoint 再导出谱掩码部署产物（BNNW 权重 + 矩阵/统计/嵌入 .h）。

用法:
  python scripts/export_mask.py --ckpt outputs/mask/checkpoints/best.pt --out outputs/mask
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

import torch

from esp_xform.mask.config import MaskConfig, _update  # type: ignore
from esp_xform.mask.export import export_masknet_artifacts
from esp_xform.mask.model import MaskNet


def _cfg_from_dict(d):
    cfg = MaskConfig()
    _update(cfg, d)
    cfg.validate()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/mask")
    ap.add_argument("--proc", default=None)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _cfg_from_dict(state["config"])
    model = MaskNet(cfg, state["num_instruments"])
    model.load_state_dict(state["model_state"])
    model.eval()

    proc = args.proc or cfg.paths.proc_dir
    out = export_masknet_artifacts(model, cfg, args.out, state["instruments"], proc_dir=proc)
    print("[export_mask] done:", out)


if __name__ == "__main__":
    main()
