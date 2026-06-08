"""MaskNet checkpoint → xform_model.bin 直接导出。

mel_mean / mel_std 优先从检查点加载（train.py 已随 best.pt 一起保存），
如果不存在则尝试从 --proc-dir 读取 stats.npz，最后回退到全零/全一占位。

用法（从项目根目录运行，或通过 export_model.bat 调用）:
  python scripts/export_bin.py \\
      --model  outputs/mask/checkpoints/best.pt \\
      --out    xform_model.bin
  python scripts/export_bin.py \\
      --model    outputs/mask/checkpoints/best.pt \\
      --out      outputs/mask/exports/xform_model.bin \\
      --proc-dir dataset/proc
"""

import _paths  # noqa: F401  — 把 src/ 加到 sys.path

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from esp_xform.mask.config import MaskConfig, _update
from esp_xform.mask.export import export_model_package
from esp_xform.mask.model import MaskNet


# ── 辅助 ────────────────────────────────────────────────────────────────────

def _cfg_from_dict(d: dict) -> MaskConfig:
    cfg = MaskConfig()
    _update(cfg, d)
    cfg.validate()
    return cfg


def _mel_stats_from_proc(proc_dir, n_mels: int):
    """从预处理目录读取 stats.npz（可选）。"""
    if proc_dir is None:
        return None, None
    npz = Path(proc_dir) / "stats.npz"
    if not npz.exists():
        return None, None
    st = np.load(str(npz))
    return st["mel_mean"].astype(np.float32), st["mel_std"].astype(np.float32)


# ── 主逻辑 ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="MaskNet checkpoint → xform_model.bin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model",    required=True,
                    help="检查点路径，例如 outputs/mask/checkpoints/best.pt")
    ap.add_argument("--out",      required=True,
                    help="输出 .bin 路径，例如 xform_model.bin")
    ap.add_argument("--proc-dir", default=None,
                    help="预处理目录（含 stats.npz）；缺省时从检查点读取梅尔统计")
    ap.add_argument("--device",   default="cpu",
                    help="推理设备（cpu / cuda）；导出只需 cpu")
    args = ap.parse_args()

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"[错误] 检查点不存在: {ckpt_path}", file=sys.stderr)
        return 1

    # ── 加载检查点 ─────────────────────────────────────────────────────────
    print(f"[export_bin] 加载检查点 …  {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    cfg      = _cfg_from_dict(ckpt["config"])
    num_inst = int(ckpt.get("num_instruments", 1))
    instr    = ckpt.get("instruments", {f"inst{i}": i for i in range(num_inst)})

    model = MaskNet(cfg, num_inst)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── 梅尔归一化统计（优先级：checkpoint > proc_dir > 占位零均值/单位方差）──
    mel_mean = ckpt.get("mel_mean", None)
    mel_std  = ckpt.get("mel_std",  None)
    if mel_mean is None or mel_std is None:
        mel_mean, mel_std = _mel_stats_from_proc(args.proc_dir, cfg.stft.n_mels)
    if mel_mean is None:
        print("[警告] 未找到梅尔统计，使用占位零均值/单位方差（推理精度可能略降）")
        mel_mean = np.zeros(cfg.stft.n_mels, np.float32)
        mel_std  = np.ones( cfg.stft.n_mels, np.float32)
    mel_mean = np.asarray(mel_mean, dtype=np.float32)
    mel_std  = np.asarray(mel_std,  dtype=np.float32)

    # ── 打印摘要 ───────────────────────────────────────────────────────────
    print(f"[export_bin] 配置: sr={cfg.stft.sample_rate}  n_fft={cfg.stft.n_fft}  "
          f"n_mels={cfg.stft.n_mels}  hidden={cfg.model.hidden}")
    print(f"[export_bin] 乐器 ({num_inst}): {list(instr.keys())}")
    print(f"[export_bin] 参数量: {model.num_parameters():,}")

    # ── 导出 xform_model.bin ───────────────────────────────────────────────
    out_path = Path(args.out)
    print(f"[export_bin] 导出 → {out_path.resolve()} …")
    result = export_model_package(model, cfg, mel_mean, mel_std, instr, out_path)

    size_kb = int(result["bytes"]) / 1024
    print(f"[export_bin] 完成: {result['model_package']}")
    print(f"[export_bin]   大小     : {size_kb:.1f} KB")
    print(f"[export_bin]   乐器数   : {result['num_instruments']}")
    print("=" * 56)
    print("  下一步：将 .bin 复制到 MCU SD 卡，重新启动固件即可。")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
