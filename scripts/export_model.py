"""从 checkpoint 重新导出产物（嵌入、归一化参数、C 头文件、配置快照）。

通常 train.py 末尾已自动导出；本脚本用于单独从某个 checkpoint 再导出。

用法：
  python scripts/export_model.py --ckpt outputs/runs/demo_run/checkpoints/best.pt \
      --out outputs/runs/demo_run
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

import torch

from esp_xform.infer import load_model
from esp_xform.train.export import export_artifacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="导出根目录；缺省为 checkpoint 所在 run 目录")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg, mean, std, instruments, _ = load_model(args.ckpt, args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if mean is None or std is None:
        raise SystemExit("该 checkpoint 未包含 feature_mean/std，请用 train.py 产出的 checkpoint。")

    out_dir = Path(args.out) if args.out else Path(args.ckpt).resolve().parents[1]
    paths = export_artifacts(model, cfg, mean, std, instruments, out_dir,
                             extra={"source_ckpt": str(args.ckpt), "metrics": ckpt.get("metrics", {})})
    print("导出完成：")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
