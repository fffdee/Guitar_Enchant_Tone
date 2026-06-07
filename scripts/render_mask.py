"""用谱掩码 checkpoint 渲染吉他音频为目标乐器（含乐器切换）。

用法:
  python scripts/render_mask.py --ckpt outputs/mask/checkpoints/best.pt \
      --source dataset/raw/clip_0001/guitar.wav --instrument piano --out out.wav
  python scripts/render_mask.py --ckpt ... --source ... --switch piano,strings,organ --out switch.wav
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

from esp_xform.audio.io import load_wav, save_wav
from esp_xform.mask.infer import load_mask_model, render_instrument, render_switch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="源吉他 wav")
    ap.add_argument("--instrument", default=None, help="单目标乐器名")
    ap.add_argument("--switch", default=None, help="逗号分隔的乐器名列表(切换演示)")
    ap.add_argument("--switch-seconds", type=float, default=2.0)
    ap.add_argument("--out", default="render_mask.wav")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="移频/变调(半音)，把输出移到目标乐器音区；0=不变调，建议 ±12 内")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="输出增益(dB)，作为驱动量；0=不变")
    ap.add_argument("--clip", default="limit", choices=["limit", "soft", "hard"],
                    help="末级削波模式：limit(干净限幅)/soft(软饱和加谐波)/hard(硬削波)")
    args = ap.parse_args()

    model, cfg, mats, mean, std, instruments = load_mask_model(args.ckpt, args.device)
    audio, _ = load_wav(args.source, expected_sr=cfg.stft.sample_rate, mono=True)
    add_noise = not args.no_noise

    if args.switch:
        names = [n.strip() for n in args.switch.split(",")]
        ids = [int(instruments[n]) for n in names]
        y, _ = render_switch(model, cfg, mats, mean, std, audio, ids,
                             args.device, args.switch_seconds, add_noise,
                             pitch_semitones=args.pitch, gain_db=args.gain,
                             clip_mode=args.clip)
    else:
        name = args.instrument or next(iter(instruments))
        y = render_instrument(model, cfg, mats, mean, std, audio,
                              int(instruments[name]), args.device, add_noise,
                              pitch_semitones=args.pitch, gain_db=args.gain,
                              clip_mode=args.clip)

    save_wav(args.out, y, cfg.stft.sample_rate)
    print(f"[render_mask] wrote {args.out} ({len(y)} samples)")


if __name__ == "__main__":
    main()
