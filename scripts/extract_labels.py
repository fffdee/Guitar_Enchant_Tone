"""对每个 (源样本, 目标乐器) 配对提取 DDSP 标签，并与源特征合并为训练样本 npz。

复用源样本缓存的逐帧 f0（平行语料音高一致）来定位目标谐波。
输出：data/processed/labels/<sample_id>__<target>.npz
  features:(T,20), labels:(T,40), f0_hz:(T,), instrument_id:(), sample_id:str

用法：python scripts/extract_labels.py
"""

import _paths  # noqa: F401

import argparse
import csv
from pathlib import Path

import numpy as np

from esp_xform.audio.io import load_wav
from esp_xform.config import load_config, load_instruments
from esp_xform.ddsp.analysis import DDSPAnalyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ddsp.yaml")
    ap.add_argument("--instruments", default="configs/instruments.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inst_cfg = load_instruments(args.instruments)
    inst_ids = inst_cfg["instruments"]

    manifest_path = Path(cfg.paths.manifest_path)
    feat_dir = Path(cfg.paths.features_dir)
    label_dir = Path(cfg.paths.labels_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise SystemExit(f"未找到 {manifest_path}，请先运行 prepare_manifest.py")

    analyzer = DDSPAnalyzer(cfg)
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_ok = 0
    for r in rows:
        sample_id = r["sample_id"]
        target = r["target_instrument"]
        feat_path = feat_dir / f"{sample_id}.npz"
        if not feat_path.exists():
            print(f"  [跳过] 缺少特征缓存 {feat_path}，请先运行 extract_features.py")
            continue

        fdata = np.load(feat_path)
        features = fdata["features"].astype(np.float32)
        f0_hz = fdata["f0_hz"].astype(np.float32)

        tgt_audio, _ = load_wav(r["target_wav"], expected_sr=cfg.audio.sample_rate)
        res = analyzer.analyze(tgt_audio, f0_hz)
        labels = res["labels"]

        # 对齐特征与标签帧数（取较短者）
        T = min(features.shape[0], labels.shape[0], f0_hz.shape[0])
        if T == 0:
            print(f"  [跳过] {sample_id}__{target} 帧数为 0")
            continue
        features, labels, f0c = features[:T], labels[:T], f0_hz[:T]

        if target not in inst_ids:
            print(f"  [警告] 乐器 {target} 不在 instruments 映射，跳过")
            continue
        instrument_id = np.int64(inst_ids[target])

        out_path = label_dir / f"{sample_id}__{target}.npz"
        np.savez(
            out_path,
            features=features,
            labels=labels,
            f0_hz=f0c,
            instrument_id=instrument_id,
            sample_id=sample_id,
        )
        n_ok += 1
        print(f"  {sample_id}__{target}: T={T}  harm_mean={labels[:, :cfg.ddsp.n_harmonics].mean():.4f}")

    print(f"完成：生成 {n_ok} 个训练样本到 {label_dir}")


if __name__ == "__main__":
    main()
