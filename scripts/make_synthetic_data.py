"""生成合成平行语料：同一段“MIDI”用不同音色渲染为 guitar/bass/piano/violin。

用途：在没有真实 Ample Sound 渲染前，提供音高/时序严格对齐的平行数据，
端到端验证训练框架（特征→标签→训练→合成→切换）。各乐器仅音色不同、音高一致，
契合“吉他→其他乐器音色转换（不改音高）”的任务设定。

用法：
  python scripts/make_synthetic_data.py --songs 8 --seconds 6
"""

import _paths  # noqa: F401  -- 注入 src 到 sys.path

import argparse
from pathlib import Path

import numpy as np

from esp_xform.audio.io import save_wav
from esp_xform.audio.normalize import peak_normalize
from esp_xform.config import load_config, load_instruments

# 各乐器音色配置（同一音高下的谐波分布/包络/噪声/揉弦差异）
PROFILES = {
    "guitar": dict(n_harm=18, decay=1.0, env="pluck", attack=0.005, tau=0.45, noise=0.010, vibrato=0.0),
    "bass":   dict(n_harm=10, decay=1.6, env="pluck", attack=0.008, tau=0.85, noise=0.006, vibrato=0.0),
    "piano":  dict(n_harm=22, decay=0.9, env="piano", attack=0.003, tau=0.35, noise=0.012, vibrato=0.0),
    "violin": dict(n_harm=26, decay=0.7, env="sustain", attack=0.060, tau=0.0, noise=0.020, vibrato=0.006),
}

_SCALE = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67]  # C 大调音阶（MIDI）


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def make_notes(rng: np.random.Generator, total_seconds: float):
    notes, acc = [], 0.0
    while acc < total_seconds:
        midi = int(rng.choice(_SCALE)) + int(rng.choice([0, 12]))
        dur = float(rng.uniform(0.35, 0.7))
        notes.append((midi, dur))
        acc += dur
    return notes


def make_env(kind: str, n: int, sr: int, attack: float, tau: float) -> np.ndarray:
    env = np.ones(n, dtype=np.float64)
    t = np.arange(n) / sr
    a = max(1, int(attack * sr))
    a = min(a, n)
    env[:a] = np.linspace(0.0, 1.0, a)
    if kind in ("pluck", "piano"):
        env *= np.exp(-t / max(tau, 1e-3))
    elif kind == "sustain":
        r = max(1, int(0.08 * sr))
        r = min(r, n)
        env[-r:] *= np.linspace(1.0, 0.0, r)
    return env


def render_note(midi: float, dur: float, sr: int, prof: dict, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(n) / sr
    f0 = midi_to_hz(midi)
    if prof["vibrato"] > 0:
        f0_t = f0 * (1.0 + prof["vibrato"] * np.sin(2 * np.pi * 5.0 * t))
    else:
        f0_t = np.full(n, f0)
    phase = np.cumsum(2 * np.pi * f0_t / sr)

    nyq = sr / 2.0
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, prof["n_harm"] + 1):
        if k * f0 >= nyq:
            break
        sig += (k ** (-prof["decay"])) * np.sin(k * phase)

    env = make_env(prof["env"], n, sr, prof["attack"], prof["tau"])
    sig *= env
    if prof["noise"] > 0:
        sig += prof["noise"] * env * rng.standard_normal(n)
    return sig


def render_instrument(notes, sr: int, prof: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = [render_note(m, d, sr, prof, rng) for m, d in notes]
    audio = np.concatenate(parts) if parts else np.zeros(0)
    return peak_normalize(audio.astype(np.float32), 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ddsp.yaml")
    ap.add_argument("--instruments", default="configs/instruments.yaml")
    ap.add_argument("--out", default="data/raw_wav")
    ap.add_argument("--songs", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sr = cfg.audio.sample_rate
    inst_map = load_instruments(args.instruments)
    all_inst = list(PROFILES.keys())          # 始终渲染 guitar + 3 目标

    out_root = Path(args.out)
    rng = np.random.default_rng(args.seed)
    n_written = 0
    for i in range(args.songs):
        notes = make_notes(rng, args.seconds)
        song = f"song_{i:03d}"
        for inst in all_inst:
            audio = render_instrument(notes, sr, PROFILES[inst], seed=args.seed + i * 17 + hash(inst) % 1000)
            path = out_root / inst / f"{song}.wav"
            save_wav(path, audio, sr)
            n_written += 1
        print(f"  {song}: {len(notes)} notes, {len(all_inst)} instruments")
    print(f"完成：共写出 {n_written} 个 WAV 到 {out_root}/<instrument>/  (采样率 {sr} Hz)")


if __name__ == "__main__":
    main()
