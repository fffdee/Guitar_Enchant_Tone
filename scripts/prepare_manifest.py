"""扫描 data/raw_wav/，按文件名配对 源吉他↔各目标乐器，生成 manifest.csv。

约定：data/raw_wav/guitar/<name>.wav 为源；data/raw_wav/<target>/<name>.wav 为目标。
仅当目标乐器存在同名文件时才生成配对行。

用法：python scripts/prepare_manifest.py
"""

import _paths  # noqa: F401

import argparse
import csv
from pathlib import Path

from esp_xform.config import load_config, load_instruments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ddsp.yaml")
    ap.add_argument("--instruments", default="configs/instruments.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inst_cfg = load_instruments(args.instruments)
    targets = inst_cfg["target_instruments"]

    raw_dir = Path(cfg.paths.raw_wav_dir)
    guitar_dir = raw_dir / "guitar"
    if not guitar_dir.exists():
        raise SystemExit(f"未找到源目录 {guitar_dir}，请先准备数据或运行 make_synthetic_data.py")

    manifest_path = Path(cfg.paths.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sorted(guitar_dir.glob("*.wav")):
        sample_id = src.stem
        for tgt in targets:
            tgt_wav = raw_dir / tgt / f"{sample_id}.wav"
            if tgt_wav.exists():
                rows.append({
                    "sample_id": sample_id,
                    "source_wav": str(src).replace("\\", "/"),
                    "target_wav": str(tgt_wav).replace("\\", "/"),
                    "target_instrument": tgt,
                })
            else:
                print(f"  [跳过] 缺少目标 {tgt_wav}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "source_wav", "target_wav", "target_instrument"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"写出 manifest：{manifest_path}（{len(rows)} 行，{len(set(r['sample_id'] for r in rows))} 个源样本）")


if __name__ == "__main__":
    main()
