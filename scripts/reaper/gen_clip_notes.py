"""生成单个 clip 的 MIDI 音符 JSON（对应 IMPLEMENTATION.md §1.3 覆盖）。

输出可直接喂给 Reaper MCP 的 midi_insert_notes_batch 的 `notes` 参数。
支持模式 --mode:
  mono_asc  单音, 从低到高 半音上行 (用户要求的"单音从低到高")
  mono_sus  单音, 长持续音 (利于音色学习), 上行
  intervals 双音音程 (三/四/五/八度)
  chords    和弦 (大三/小三/属七/小七/挂四/强力)
  mixed     单音/音程/和弦 随机混合
用法: python gen_clip_notes.py --mode mono_asc --seconds 15.5 --out notes.json
"""
from __future__ import annotations
import argparse, json, random

CHORDS = [[0, 4, 7], [0, 3, 7], [0, 4, 7, 10], [0, 3, 7, 10], [0, 5, 7], [0, 7]]
INTERVALS = [3, 4, 5, 7, 12]
LO, HI = 40, 76          # E2..E5 吉他常用音域
VELS = [70, 90, 110]


def _emit(notes, pitches, t, dur, vel, max_notes):
    for p in pitches:
        if 28 <= p <= 96 and len(notes) < max_notes:
            notes.append({"pitch": p, "velocity": vel,
                          "start": round(t, 4), "end": round(t + dur, 4), "channel": 0})


def gen(mode: str, seconds: float, seed: int, max_notes: int):
    rng = random.Random(seed)
    notes = []
    t = 0.2
    if mode in ("mono_asc", "mono_sus"):
        dur, gap, step = (0.40, 0.06, 1) if mode == "mono_asc" else (0.90, 0.12, 2)
        p = LO
        vi = 0
        while p <= HI and t + dur <= seconds and len(notes) < max_notes:
            _emit(notes, [p], t, dur, VELS[vi % len(VELS)], max_notes)
            t += dur + gap
            p += step
            vi += 1
    elif mode == "intervals":
        root = LO
        while t + 0.6 <= seconds and len(notes) < max_notes and root <= HI:
            iv = INTERVALS[(root - LO) % len(INTERVALS)]
            dur = 0.6
            _emit(notes, [root, root + iv], t, dur, VELS[(root) % len(VELS)], max_notes)
            t += dur + 0.1
            root += 3
    elif mode == "chords":
        root = LO
        ci = 0
        while t + 0.7 <= seconds and len(notes) < max_notes and root <= HI - 7:
            _emit(notes, [root + o for o in CHORDS[ci % len(CHORDS)]], t, 0.7,
                  VELS[ci % len(VELS)], max_notes)
            t += 0.7 + 0.15
            root += 4
            ci += 1
    else:  # mixed
        durs = [0.3, 0.5, 0.8, 1.2]
        rests = [0.0, 0.1, 0.2]
        while t < seconds - 0.3 and len(notes) < max_notes:
            kind = rng.randint(1, 3)
            root = LO + rng.randint(0, HI - LO)
            if kind == 1:
                pitches = [root]
            elif kind == 2:
                pitches = [root, root + rng.choice(INTERVALS)]
            else:
                pitches = [root + o for o in rng.choice(CHORDS)]
            dur = rng.choice(durs)
            if t + dur > seconds:
                dur = seconds - t
            _emit(notes, pitches, t, dur, rng.choice(VELS), max_notes)
            t += dur + rng.choice(rests)
    end = max((n["end"] for n in notes), default=seconds)
    return notes, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mono_asc",
                    choices=["mono_asc", "mono_sus", "intervals", "chords", "mixed"])
    ap.add_argument("--seconds", type=float, default=15.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-notes", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    notes, end = gen(args.mode, args.seconds, args.seed, args.max_notes)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(notes, f, separators=(",", ":"))
    print(f"mode={args.mode} notes={len(notes)} duration={end:.3f}")


if __name__ == "__main__":
    main()
