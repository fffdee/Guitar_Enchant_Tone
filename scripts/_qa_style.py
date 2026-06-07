import sys, os, glob, json
sys.path.insert(0, "src")
import numpy as np
from esp_xform.audio.io import load_wav

root = sys.argv[1] if len(sys.argv) > 1 else "dataset/raw_style"
n_ok = 0
for clip in sorted(glob.glob(os.path.join(root, "clip_*"))):
    wavs = sorted(os.path.basename(w) for w in glob.glob(os.path.join(clip, "*.wav")))
    meta = {}
    mp = os.path.join(clip, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
    parts, lens = [], {}
    for w in wavs:
        y, sr = load_wav(os.path.join(clip, w), expected_sr=None, mono=True)
        pk = float(np.abs(y).max()); rms = float(np.sqrt(np.mean(y ** 2)))
        flag = " SILENT!" if pk < 1e-3 else ""
        parts.append("%s sr=%d n=%d pk=%.3f rms=%.4f%s" % (w[:-4], sr, len(y), pk, rms, flag))
        lens[w] = len(y)
    tgt = (meta.get("targets") or ["?"])[0]
    aligned = len(set(lens.values())) <= 1
    if aligned and all("SILENT" not in p for p in parts):
        n_ok += 1
    print("%s tgt=%-8s pitchD=%s aligned=%s | %s" % (
        os.path.basename(clip), tgt, meta.get("src_minus_tgt_first_pitch", "?"),
        aligned, "  ||  ".join(parts)))
print("\n[QA] %d clips, %d ok (aligned & non-silent)" % (
    len(glob.glob(os.path.join(root, "clip_*"))), n_ok))
