"""把 Reaper MCP 渲染出的 wav（默认在 REAPER Media，44.1k/立体声）
规整为项目要求的 48k 单声道并落到 dataset 目标路径。

用法:
  python ingest_render.py --src-dir "C:/Users/BanGO/Documents/REAPER Media" \
      --dst dataset/raw/clip_0001/guitar.wav --sr 48000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from esp_xform.audio.io import load_wav, save_wav
from esp_xform.audio.resample import resample_to


def newest_wav(d: Path) -> Path:
    wavs = sorted(d.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise FileNotFoundError(f"no .wav in {d}")
    return wavs[0]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", help="指定源 wav")
    g.add_argument("--src-dir", help="取该目录下最新的 wav")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()

    import numpy as np
    src = Path(args.src) if args.src else newest_wav(Path(args.src_dir))
    audio, sr = load_wav(src, expected_sr=None, mono=True)
    y = resample_to(audio, sr, args.sr)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_wav(dst, y, args.sr)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    verdict = "SILENT!!" if peak < 1e-3 else "ok"
    print(f"[ingest] {src.name} {sr}Hz -> {dst} {args.sr}Hz {len(y)}smp "
          f"peak={peak:.4f} rms={rms:.5f} [{verdict}]")


if __name__ == "__main__":
    main()
