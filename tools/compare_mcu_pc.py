"""MCU vs PC 谱掩码推理数值对拍与参数对齐工具.

Task 1 — 参数对齐:
  MCU CLI 默认: gain=0 dB, clip=limit, add_noise=0 (infer 无 -g/-c/-N)
  PC 对照渲染应使用相同参数, 再与带后处理的 guitar__to_bass_g+9_soft.wav 对比.

Task 2 — 分层对拍:
  模拟 MCU BLOCK_FRAMES=64 + CTX=4 的分块 CNN, 输出 logmel/mask/gain_lin 供与串口 debug 日志对比.

用法:
  python tools/compare_mcu_pc.py --ckpt outputs/mask_style/checkpoints/best.pt \\
      --source dataset/raw_style/clip_0001/guitar.wav --instrument bass

  python tools/compare_mcu_pc.py ... --render-baseline   # 生成 MCU 默认参数参考 wav
  python tools/compare_mcu_pc.py ... --mcu-log mcu_debug.txt  # 解析 MCU debug 日志
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BLOCK_FRAMES = 64
CTX_FRAMES = 4

MCU_DEFAULTS = {
    "gain_db": 0.0,
    "clip_mode": "limit",
    "add_noise": False,
    "pitch_semitones": 0.0,
}


def _parse_mcu_debug_log(text: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    patterns = {
        "logmel": r"debug block0 frame0 logmel\[0:8\]:\s*([-\d.e+]+(?:\s+[-\d.e+]+){7})",
        "mask": r"debug block0 frame0 mask\[0:8\]:\s*([-\d.e+]+(?:\s+[-\d.e+]+){7})",
        "gain_lin": r"debug block0 frame0 gain_lin\[0:8\]:\s*([-\d.e+]+(?:\s+[-\d.e+]+){7})",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = np.array([float(x) for x in m.group(1).split()], dtype=np.float64)
    return out


def _blocked_predict_heads(model, x_full: np.ndarray, inst_id: int,
                           block: int = BLOCK_FRAMES, ctx: int = CTX_FRAMES,
                           device: str = "cpu"):
    """模拟 MCU 分块 CNN (含跨块 CTX 前缀), 返回与整段等长的三头输出."""
    import torch

    was_training = model.training
    model.eval()
    x = x_full[0]  # [M, T]
    M, T = x.shape
    P, B = model.phase_bands, model.noise_bands
    mask_out = np.zeros((M, T), np.float32)
    dphi_out = np.zeros((P, T), np.float32)
    noise_out = np.zeros((B, T), np.float32)

    in0 = np.zeros((M, block), np.float32)
    have_ctx = False
    ctx_buf = np.zeros((M, ctx), np.float32)
    chunk = 0
    inst = torch.tensor([inst_id], dtype=torch.long, device=device)

    def _run_block(cols: int, t_out0: int):
        xt = in0[:, :cols].copy()
        if cols < block:
            pad = np.zeros((M, block - cols), np.float32)
            xt = np.concatenate([xt, pad], axis=1)
        xt = torch.tensor(xt[None].astype(np.float32), device=device)
        with torch.no_grad():
            mk, dp, ns = model(xt, inst)
        n = min(cols, T - t_out0)
        mask_out[:, t_out0:t_out0 + n] = mk[0, :, :n].detach().cpu().numpy()
        dphi_out[:, t_out0:t_out0 + n] = dp[0, :, :n].detach().cpu().numpy()
        noise_out[:, t_out0:t_out0 + n] = ns[0, :, :n].detach().cpu().numpy()

    for t in range(T):
        write_col = (ctx + chunk) if have_ctx else chunk
        in0[:, write_col] = x[:, t]
        chunk += 1
        target = (block - ctx) if have_ctx else block
        if chunk == target:
            if have_ctx:
                in0[:, :ctx] = ctx_buf
            t_out0 = t - target + 1
            _run_block(block, t_out0)
            ctx_buf = in0[:, block - ctx:block].copy()
            have_ctx = True
            chunk = 0
            in0[:] = 0.0

    if chunk > 0:
        cols = (ctx + chunk) if have_ctx else chunk
        if have_ctx:
            in0[:, :ctx] = ctx_buf
        t_out0 = T - chunk
        _run_block(cols, t_out0)

    return mask_out, dphi_out, noise_out


def _analyze_like_mcu(cfg, mats, mel_mean, mel_std, audio: np.ndarray):
    """STFT + logmel, 与 infer._analyze 一致 (float64)."""
    from esp_xform.mask.audio import stft

    s = cfg.stft
    Xc = stft(audio, s.n_fft, s.hop, center=s.center)
    Xlin_t = np.abs(Xc)                     # [T, n_bins]
    phase_t = np.angle(Xc)
    mel_basis = mats["mel_basis"].astype(np.float64)
    Xmel = Xlin_t @ mel_basis.T               # [T, M]
    logmel = np.log(Xmel + s.log_eps)
    if mel_mean is not None and mel_std is not None:
        logmel = (logmel - np.asarray(mel_mean)) / (np.asarray(mel_std) + 1e-8)
    x = logmel.T[None].astype(np.float32)
    return Xlin_t.T.copy(), phase_t.T.copy(), x


def _print_vec(name: str, v: np.ndarray, ref: np.ndarray | None = None):
    print(f"  {name}[0:8]: " + " ".join(f"{x:+.4f}" for x in v[:8]))
    if ref is not None and ref.size >= 8:
        d = v[:8] - ref[:8]
        print(f"  {name} delta: " + " ".join(f"{x:+.4f}" for x in d))
        print(f"  {name} max_abs_delta[0:8] = {np.max(np.abs(d)):.6f}")


def main():
    ap = argparse.ArgumentParser(description="MCU vs PC 推理对拍")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", default=str(ROOT / "dataset/raw_style/clip_0002/guitar.wav"),
                    help="源吉他 wav (默认 clip_0002, 与常见 mcu_bass 时长 ~19.6s 对齐)")
    ap.add_argument("--instrument", default="bass")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--render-baseline", action="store_true",
                    help="用 MCU 默认参数渲染参考 wav (gain=0, limit, no-noise)")
    ap.add_argument("--out", default=str(ROOT / "outputs/mask_style/renders/guitar__to_bass_mcu_baseline.wav"))
    ap.add_argument("--mcu-log", default=None, help="MCU debug 命令输出的日志文件")
    ap.add_argument("--compare-styled", default=str(ROOT / "outputs/mask_style/renders/guitar__to_bass_g+9_soft.wav"),
                    help="带后处理的 PC 参考 (用于说明参数差异)")
    args = ap.parse_args()

    from esp_xform.audio.io import load_wav, save_wav
    from esp_xform.mask.infer import load_mask_model, render_instrument
    from esp_xform.mask.reconstruct import reconstruct_np

    model, cfg, mats, mean, std, instruments = load_mask_model(args.ckpt, args.device)
    audio, _ = load_wav(args.source, expected_sr=cfg.stft.sample_rate, mono=True)
    inst_id = int(instruments[args.instrument])

    print("=== Task 1: 参数对齐 ===")
    print(f"MCU 默认: gain={MCU_DEFAULTS['gain_db']} dB, clip={MCU_DEFAULTS['clip_mode']}, "
          f"add_noise={MCU_DEFAULTS['add_noise']}, pitch={MCU_DEFAULTS['pitch_semitones']}")
    print(f"PC styled 参考: {args.compare_styled}")
    print("  (g+9_soft = gain +9dB + tanh 软饱和, 与 MCU 默认差异最大)\n")

    if args.render_baseline:
        y = render_instrument(model, cfg, mats, mean, std, audio, inst_id, args.device,
                              add_noise=MCU_DEFAULTS["add_noise"],
                              pitch_semitones=MCU_DEFAULTS["pitch_semitones"],
                              gain_db=MCU_DEFAULTS["gain_db"],
                              clip_mode=MCU_DEFAULTS["clip_mode"])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_wav(str(out), y, cfg.stft.sample_rate)
        print(f"已写入 MCU 对齐基准: {out} ({len(y)} samples)\n")

    print("=== Task 2: 分层数值 (block0 frame0) ===")
    print("健康 MCU (F32): mask mean≈3.5~4.0; 失效 (INT8/零权重): mask mean≈2.0, noise_t 全 0.693\n")
    Xlin, phase, x = _analyze_like_mcu(cfg, mats, mean, std, audio)
    logmel0 = x[0, :, 0].astype(np.float64)

    mask_full, dphi_full, noise_full = _blocked_predict_heads(
        model, x, inst_id, device=args.device)
    mask0 = mask_full[:, 0]
    gain_lin0 = mats["mel_inv"].astype(np.float64) @ mask0.astype(np.float64)

    print("PC (整段 analyze + 分块 CNN block0/frame0):")
    _print_vec("logmel", logmel0)
    _print_vec("mask", mask0)
    _print_vec("gain_lin", gain_lin0)

    if args.mcu_log:
        text = Path(args.mcu_log).read_text(encoding="utf-8", errors="replace")
        mcu = _parse_mcu_debug_log(text)
        if not mcu:
            print(f"\n未在 {args.mcu_log} 中找到 debug 行 (先运行: debug bass -i /sdcard/in/guitar.wav)")
        else:
            print(f"\nMCU 日志 ({args.mcu_log}) vs PC:")
            if "logmel" in mcu:
                _print_vec("logmel", logmel0, mcu["logmel"])
            if "mask" in mcu:
                _print_vec("mask", mask0, mcu["mask"])
            if "gain_lin" in mcu:
                _print_vec("gain_lin", gain_lin0, mcu["gain_lin"])

    # 全链路 MCU 默认渲染 vs 分块重建 (无后处理)
    y_blocked = reconstruct_np(
        Xlin, phase, mask_full, dphi_full, noise_full,
        mats["mel_inv"], mats["phase_inv"], mats["noise_fb"],
        cfg.stft.n_fft, cfg.stft.hop, center=cfg.stft.center,
        add_noise=MCU_DEFAULTS["add_noise"],
    )
    y_full = render_instrument(model, cfg, mats, mean, std, audio, inst_id, args.device,
                               add_noise=MCU_DEFAULTS["add_noise"],
                               gain_db=0.0, clip_mode="limit")
    n = min(len(y_blocked), len(y_full))
    diff = y_blocked[:n].astype(np.float64) - y_full[:n].astype(np.float64)
    print(f"\n分块 CNN 重建 vs 整段推理 (MCU 默认后处理):")
    print(f"  max_abs_diff={np.max(np.abs(diff)):.6f}  RMS diff={np.sqrt(np.mean(diff**2)):.6f}")

    print("\n下一步:")
    print("  1) 烧录固件后在 MCU 运行: debug bass -i /sdcard/in/guitar.wav")
    print("  2) 保存串口日志, 再运行: python tools/compare_mcu_pc.py ... --mcu-log debug.txt")


if __name__ == "__main__":
    main()
