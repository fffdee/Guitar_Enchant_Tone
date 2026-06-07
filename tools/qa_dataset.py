"""数据集质检 (对应 IMPLEMENTATION.md §9 / 工程说明 §9.2 tools/qa_dataset.py)。

扫描 dataset/raw/clip_*/，逐 clip 检查 guitar/目标 wav 的采样率、声道、时长、
峰值/RMS(静音判定)、源/目标长度对齐差。供上位机 GUI 调用，也可单独运行。

用法: python tools/qa_dataset.py --raw dataset/raw --sr 48000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from esp_xform.audio.io import load_wav


def _stat(path: Path):
    audio, sr = load_wav(path, expected_sr=None, mono=True)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    return {"sr": sr, "n": len(audio), "sec": len(audio) / sr if sr else 0.0,
            "peak": peak, "rms": rms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--source", default="guitar", help="源(输入)文件名(不含 .wav)")
    args = ap.parse_args()

    raw = Path(args.raw)
    clips = sorted([d for d in raw.glob("clip_*") if d.is_dir()])
    if not clips:
        print(f"[qa] 未在 {raw} 找到 clip_* 目录")
        return

    n_ok, n_warn = 0, 0
    print(f"[qa] 扫描 {raw}  期望采样率={args.sr}Hz  源={args.source}\n")
    print(f"{'clip':<12}{'file':<10}{'sr':>7}{'sec':>8}{'peak':>9}{'rms':>9}  flags")
    print("-" * 70)
    for d in clips:
        src = d / f"{args.source}.wav"
        wavs = sorted(d.glob("*.wav"))
        if not wavs:
            print(f"{d.name:<12}(空: 无 wav)")
            n_warn += 1
            continue
        src_sec = None
        for w in wavs:
            try:
                st = _stat(w)
            except Exception as e:
                print(f"{d.name:<12}{w.stem:<10}  读取失败: {type(e).__name__} (可能被 Reaper 占用)")
                n_warn += 1
                continue
            flags = []
            if st["sr"] != args.sr:
                flags.append(f"SR!={args.sr}")
            if st["peak"] < 1e-3:
                flags.append("静音")
            if w.name == src.name:
                src_sec = st["sec"]
            elif src_sec is not None and abs(st["sec"] - src_sec) > 0.05:
                flags.append(f"长度差{st['sec'] - src_sec:+.2f}s")
            tag = "OK" if not flags else "  ".join(flags)
            if flags:
                n_warn += 1
            else:
                n_ok += 1
            print(f"{d.name:<12}{w.stem:<10}{st['sr']:>7}{st['sec']:>8.2f}"
                  f"{st['peak']:>9.4f}{st['rms']:>9.4f}  {tag}")
    print("-" * 70)
    print(f"[qa] clip={len(clips)}  文件OK={n_ok}  警告={n_warn}")
    if n_warn == 0:
        print("[qa] 全部通过, 可执行预处理。")


if __name__ == "__main__":
    main()
