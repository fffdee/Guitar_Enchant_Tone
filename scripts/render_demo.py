"""用训练好的模型对一段源吉他音频做转换渲染（含流式乐器切换演示）。

用法：
  python scripts/render_demo.py --ckpt outputs/runs/demo_run/checkpoints/best.pt \
      --source data/raw_wav/guitar/song_000.wav --instruments bass piano violin
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from esp_xform.audio.io import load_wav, save_wav
from esp_xform.infer import load_model, render_instrument, streaming_render_switch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="源吉他 WAV")
    ap.add_argument("--instruments", nargs="*", default=None, help="目标乐器名列表；缺省用全部目标")
    ap.add_argument("--out", default="outputs/renders")
    ap.add_argument("--switch-seconds", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg, mean, std, instruments, band_edges = load_model(args.ckpt, args.device)
    inst_ids = instruments.get("instruments", {})
    targets = args.instruments or instruments.get("target_instruments", list(inst_ids.keys()))

    source, sr = load_wav(args.source, expected_sr=cfg.audio.sample_rate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem

    # 1) 各目标乐器单独渲染
    for tgt in targets:
        if tgt not in inst_ids:
            print(f"  [跳过] 未知乐器 {tgt}")
            continue
        audio = render_instrument(model, source, inst_ids[tgt], cfg, mean, std, band_edges, args.device)
        path = out_dir / f"{stem}__to_{tgt}.wav"
        save_wav(path, audio, sr)
        print(f"  渲染 {tgt} -> {path}  (RMS={np.sqrt(np.mean(audio**2)):.4f})")

    # 2) 流式乐器切换渲染
    target_id_list = [inst_ids[t] for t in targets if t in inst_ids]
    if target_id_list:
        audio, sched = streaming_render_switch(
            model, source, target_id_list, cfg, mean, std, band_edges,
            args.device, switch_seconds=args.switch_seconds,
        )
        path = out_dir / f"{stem}__switch.wav"
        save_wav(path, audio, sr)
        n_switch = int(np.sum(np.diff(sched) != 0))
        print(f"  切换渲染 -> {path}  (切换 {n_switch} 次，每 {args.switch_seconds}s 轮换)")

    print("完成。")


if __name__ == "__main__":
    main()
