"""对每个源吉他样本提取 20 维特征并缓存（每个 sample_id 只算一次）。

输出：data/processed/features/<sample_id>.npz
  features:(T,20), f0_hz:(T,), voiced:(T,), loudness_db:(T,)

用法：python scripts/extract_features.py
"""

import _paths  # noqa: F401

import argparse
import csv
from pathlib import Path

import numpy as np

from esp_xform.audio.io import load_wav
from esp_xform.config import load_config
from esp_xform.features import GuitarFeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ddsp.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(cfg.paths.manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"未找到 {manifest_path}，请先运行 prepare_manifest.py")

    feat_dir = Path(cfg.paths.features_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    extractor = GuitarFeatureExtractor(cfg)

    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sources = {}
    for r in rows:
        sources[r["sample_id"]] = r["source_wav"]

    for sample_id, src_wav in sorted(sources.items()):
        audio, sr = load_wav(src_wav, expected_sr=cfg.audio.sample_rate)
        out = extractor.extract(audio)
        out_path = feat_dir / f"{sample_id}.npz"
        np.savez(
            out_path,
            features=out["features"],
            f0_hz=out["f0_hz"],
            voiced=out["voiced"],
            loudness_db=out["loudness_db"],
        )
        print(f"  {sample_id}: {out['features'].shape[0]} 帧 -> {out_path.name}")

    print(f"完成：{len(sources)} 个源样本的特征已缓存到 {feat_dir}")


if __name__ == "__main__":
    main()
