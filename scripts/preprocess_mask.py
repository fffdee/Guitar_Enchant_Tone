"""谱掩码方案数据预处理：raw/clip_*/ → 对齐波形对 + 窗口索引 + 梅尔统计 + 展开矩阵。

用法:
  python scripts/preprocess_mask.py --config configs/mask.json --instruments configs/instruments.yaml
"""

import _paths  # noqa: F401

import argparse

from esp_xform.config import load_instruments
from esp_xform.mask.config import load_mask_config
from esp_xform.mask.dataset import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="mask 配置 (yaml/json)")
    ap.add_argument("--instruments", default=None, help="乐器映射 (yaml/json)")
    ap.add_argument("--raw", default=None, help="覆盖 raw 目录")
    ap.add_argument("--proc", default=None, help="覆盖 proc 目录")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 个片段(调试)")
    args = ap.parse_args()

    cfg = load_mask_config(args.config)
    if args.raw:
        cfg.paths.raw_dir = args.raw
    if args.proc:
        cfg.paths.proc_dir = args.proc

    inst = load_instruments(args.instruments)["instruments"]
    info = preprocess(cfg, inst, limit=args.limit)
    print("[preprocess_mask] done:", info)


if __name__ == "__main__":
    main()
