"""谱掩码模型训练 + 导出。

用法:
  python scripts/train_mask.py --config configs/mask.json --epochs 200 --device cuda
"""

import _paths  # noqa: F401

import argparse

from esp_xform.config import load_instruments
from esp_xform.mask.config import load_mask_config
from esp_xform.mask.export import export_masknet_artifacts
from esp_xform.mask.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--instruments", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--proc", default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    cfg = load_mask_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.proc:
        cfg.paths.proc_dir = args.proc

    inst = load_instruments(args.instruments)["instruments"]
    info = train(cfg, inst, device=args.device, proc_dir=args.proc)
    print("[train_mask] training done:", info)

    if not args.no_export:
        import torch
        from esp_xform.mask.model import MaskNet
        from pathlib import Path

        ckpt = Path(info["ckpt_dir"]) / "best.pt"
        if not ckpt.exists():
            ckpt = Path(info["ckpt_dir"]) / "last.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = MaskNet(cfg, state["num_instruments"])
        model.load_state_dict(state["model_state"])
        model.eval()
        out = export_masknet_artifacts(model, cfg, cfg.paths.output_dir,
                                       state["instruments"], proc_dir=cfg.paths.proc_dir)
        print("[train_mask] export done:", out)


if __name__ == "__main__":
    main()
