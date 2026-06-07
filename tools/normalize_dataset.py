"""规整渲染产物: 把 dataset/raw/clip_*/<stem>.wav 统一为 48k/16bit/单声道的文件。

兼容两种来源:
  - 正常文件 clip/stem.wav             -> 原地标准化(若已是48k/mono则跳过)
  - Reaper 误当目录   clip/stem.wav/<x>.wav -> 取其中 wav 展平为 clip/stem.wav 文件
被占用(Reaper 未释放)的文件会跳过并提示。

用法: python tools/normalize_dataset.py --raw dataset/raw --sr 48000
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from esp_xform.audio.io import load_wav, save_wav
from esp_xform.audio.resample import resample_to


def _standardize(src: Path, dst: Path, sr: int) -> str:
    audio, in_sr = load_wav(src, expected_sr=None, mono=True)
    y = resample_to(audio, in_sr, sr)
    tmp = dst.with_suffix(".wav.tmp")
    save_wav(tmp, y, sr)
    return f"{in_sr}Hz {len(audio)}smp -> {sr}Hz {len(y)}smp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()
    raw = Path(args.raw)
    clips = sorted([d for d in raw.glob("clip_*") if d.is_dir()])
    if not clips:
        print(f"[normalize] 未找到 {raw}/clip_*")
        return

    fixed, skipped = 0, 0
    for d in clips:
        # 收集候选: 文件 *.wav 与 误建为目录的 *.wav
        entries = [p for p in d.iterdir() if p.name.lower().endswith(".wav")]
        for ent in entries:
            stem_path = d / ent.name  # 目标文件 clip/stem.wav
            try:
                if ent.is_dir():
                    inner = sorted(ent.glob("*.wav"), key=lambda p: p.stat().st_size, reverse=True)
                    if not inner:
                        print(f"  {d.name}/{ent.name}  (空目录, 跳过)"); skipped += 1; continue
                    info = _standardize(inner[0], stem_path, args.sr)
                    shutil.rmtree(ent)                       # 删目录
                    Path(str(stem_path) + ".tmp").replace(stem_path)
                    print(f"  {d.name}/{ent.name}  展平+规整  {info}"); fixed += 1
                else:
                    info = _standardize(ent, stem_path, args.sr)
                    Path(str(stem_path) + ".tmp").replace(stem_path)
                    print(f"  {d.name}/{ent.name}  规整  {info}"); fixed += 1
            except PermissionError:
                print(f"  {d.name}/{ent.name}  被占用(Reaper 未释放?), 跳过"); skipped += 1
            except Exception as e:
                print(f"  {d.name}/{ent.name}  失败: {type(e).__name__}: {e}"); skipped += 1

    print(f"[normalize] 完成: 规整 {fixed}  跳过 {skipped}")


if __name__ == "__main__":
    main()
