"""谱掩码模型训练（对应 §4）。混合损失：mask + logmel + 多分辨率STFT + 复数STFT。

按 pair 文件划分 train/val 防泄漏；保存 best/last checkpoint（自包含 cfg/instruments）。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import MaskConfig
from .dataset import MaskWindowDataset, MaskWaveDataset, collate
from .losses import complex_stft_loss, logmel_l1, mask_l1, multi_res_stft
from .model import MaskNet
from .reconstruct import SpectralReconstructor


def _safe_print(*args, **kwargs) -> None:
    """对 stdout 失效免疫的打印。

    GUI(上位机)常从后台终端启动，长时间运行后该终端的输出管道可能失效，
    Windows 下 print 会抛 OSError: [Errno 22] Invalid argument。训练进度本应
    通过 on_epoch 回调上报，stdout 仅为 CLI 便利，绝不能因打印失败而中断训练。
    """
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def _split_indices(dataset, val_ratio: float, seed: int):
    files = sorted({rec["file"] for rec in dataset.index})
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
    val_files = set(files[:n_val])
    train_idx, val_idx = [], []
    for i, rec in enumerate(dataset.index):
        (val_idx if rec["file"] in val_files else train_idx).append(i)
    return train_idx, val_idx


def train(cfg: MaskConfig, instruments: Dict[str, int],
          device: Optional[str] = None, proc_dir: Optional[str] = None,
          on_epoch=None, should_stop=None) -> Dict[str, str]:
    """训练谱掩码模型。

    on_epoch: 可选回调 on_epoch(ep:int, train_loss:float, val_logmel:Optional[float], best_val:float)，
              供上位机实时更新曲线（在调用线程内同步触发）。
    should_stop: 可选无参回调，返回 True 时在本轮结束后提前停止（供上位机“停止”按钮）。
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg.train
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    random.seed(tc.seed)

    aug = None
    if tc.augment:
        from .augment import AugmentConfig
        aug = AugmentConfig()
    ds = MaskWindowDataset(cfg, proc_dir, augment=aug)        # 训练集(可增强)
    ds_val = MaskWindowDataset(cfg, proc_dir, augment=None)   # 验证集(不增强)
    proc = Path(proc_dir or cfg.paths.proc_dir)
    num_inst = max(instruments.values()) + 1
    mel_mean = ds.mel_mean.astype(np.float32)
    mel_std = ds.mel_std.astype(np.float32)

    def _ckpt_payload():
        return {"model_state": model.state_dict(), "config": cfg.to_dict(),
                "instruments": instruments, "num_instruments": num_inst,
                "mel_mean": mel_mean, "mel_std": mel_std}

    train_idx, val_idx = _split_indices(ds, tc.val_ratio, tc.seed)
    dl_tr = DataLoader(Subset(ds, train_idx), batch_size=tc.batch_size, shuffle=True,
                       num_workers=tc.num_workers, collate_fn=collate, drop_last=True)
    dl_va = DataLoader(Subset(ds_val, val_idx), batch_size=tc.batch_size, shuffle=False,
                       num_workers=tc.num_workers, collate_fn=collate) if val_idx else None

    model = MaskNet(cfg, num_inst).to(device)
    mats = {k: np.load(proc / f"{k}.npy") for k in ("mel_inv", "phase_inv", "noise_fb")}
    recon = SpectralReconstructor(cfg, mats["mel_inv"], mats["phase_inv"], mats["noise_fb"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    _safe_print(f"[mask.train] device={device} params={model.num_parameters()} "
                f"train_windows={len(train_idx)} val_windows={len(val_idx)}")

    out = Path(cfg.paths.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    win_samples = tc.win_frames * cfg.stft.hop
    best_val = float("inf")

    def run_batch(b, train_mode: bool):
        for k in b:
            b[k] = b[k].to(device)
        mask, dphi, noise = model(b["x"], b["inst_id"])
        Lm = mask_l1(mask, b["mask_lab"])
        Llog = logmel_l1(mask, b["Xmel"], b["Ymel"])
        loss = tc.w_mask * Lm + tc.w_logmel * Llog
        if tc.w_mrstft > 0:
            y = recon(b["Xlin"], b["phase"], mask, dphi, noise, length=win_samples)
            loss = loss + tc.w_mrstft * multi_res_stft(y, b["target_wav"], tc.mrstft_ffts, tc.mrstft_hops)
        if tc.use_phase and tc.w_cplx > 0:
            loss = loss + tc.w_cplx * complex_stft_loss(
                b["Xlin"], b["phase"], mask, dphi, recon.mel_inv, recon.phase_inv,
                b["Ymag"], b["Yphase"])
        return loss, Llog

    for ep in range(tc.epochs):
        model.train()
        tl = 0.0
        n = 0
        for b in dl_tr:
            loss, _ = run_batch(b, True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tl += loss.item()
            n += 1
        msg = f"epoch {ep:3d} train_loss={tl / max(n, 1):.4f}"

        val_metric: Optional[float] = None
        if dl_va is not None:
            model.eval()
            vlog = 0.0
            nv = 0
            with torch.no_grad():
                for b in dl_va:
                    _, Llog = run_batch(b, False)
                    vlog += float(Llog)
                    nv += 1
            vlog /= max(nv, 1)
            val_metric = vlog
            msg += f"  val_logmel={vlog:.4f}"
            if vlog < best_val:
                best_val = vlog
                torch.save(_ckpt_payload(), ckpt_dir / "best.pt")
        print(f"[mask.train] {msg}")

        if on_epoch is not None:
            try:
                on_epoch(ep, float(tl / max(n, 1)), val_metric, float(best_val))
            except Exception:
                pass
        if should_stop is not None and should_stop():
            print("[mask.train] 收到停止请求，提前结束。")
            break

    torch.save(_ckpt_payload(), ckpt_dir / "last.pt")
    return {"ckpt_dir": str(ckpt_dir), "best_val_logmel": f"{best_val:.6f}"}


def train_gpu(cfg: MaskConfig, instruments: Dict[str, int],
              device: Optional[str] = None, proc_dir: Optional[str] = None,
              on_epoch=None, should_stop=None) -> Dict[str, str]:
    """训练谱掩码模型（GPU加速版本）。

    使用 PyTorch GPU STFT 替代 numpy STFT，大幅提升训练速度。
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg.train
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    random.seed(tc.seed)

    aug = None
    if tc.augment:
        from .augment import AugmentConfig
        aug = AugmentConfig()
    ds = MaskWaveDataset(cfg, proc_dir, augment=aug)        # 训练集(可增强)
    ds_val = MaskWaveDataset(cfg, proc_dir, augment=None)   # 验证集(不增强)
    proc = Path(proc_dir or cfg.paths.proc_dir)
    num_inst = max(instruments.values()) + 1
    mel_mean = ds.mel_mean.cpu().numpy()
    mel_std = ds.mel_std.cpu().numpy()

    def _ckpt_payload():
        return {"model_state": model.state_dict(), "config": cfg.to_dict(),
                "instruments": instruments, "num_instruments": num_inst,
                "mel_mean": mel_mean, "mel_std": mel_std}

    train_idx, val_idx = _split_indices(ds, tc.val_ratio, tc.seed)
    dl_tr = DataLoader(Subset(ds, train_idx), batch_size=tc.batch_size, shuffle=True,
                       num_workers=0, collate_fn=collate, drop_last=True)
    dl_va = DataLoader(Subset(ds_val, val_idx), batch_size=tc.batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate) if val_idx else None

    model = MaskNet(cfg, num_inst).to(device)
    mats = {k: np.load(proc / f"{k}.npy") for k in ("mel_inv", "phase_inv", "noise_fb")}
    recon = SpectralReconstructor(cfg, mats["mel_inv"], mats["phase_inv"], mats["noise_fb"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    
    mel_basis = ds.mel_basis.to(device)
    mel_mean = ds.mel_mean.to(device)
    mel_std = ds.mel_std.to(device)
    window = ds._window.to(device)
    n_fft = cfg.stft.n_fft
    hop = cfg.stft.hop
    log_eps = cfg.stft.log_eps
    mask_eps = cfg.stft.mask_eps
    gmax = cfg.model.gmax
    T = cfg.train.win_frames

    _safe_print(f"[mask.train_gpu] device={device} params={model.num_parameters()} "
                f"train_windows={len(train_idx)} val_windows={len(val_idx)}")

    out = Path(cfg.paths.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    win_samples = tc.win_frames * cfg.stft.hop
    best_val = float("inf")

    def run_batch(b, train_mode: bool):
        g_wav = b["g_wav"].to(device)      # [B, win_samples]
        t_wav = b["t_wav"].to(device)      # [B, win_samples]
        inst_id = b["inst_id"].to(device)   # [B]

        Xc = torch.stft(g_wav, n_fft, hop, window=window, center=True,
                        return_complex=True, pad_mode="reflect")  # [B, n_bins, T, complex]
        Tc = torch.stft(t_wav, n_fft, hop, window=window, center=True,
                        return_complex=True, pad_mode="reflect")
        Xc = Xc[:, :, :T]                  # [B, n_bins, T]
        Tc = Tc[:, :, :T]

        Xlin = Xc.abs()                    # [B, n_bins, T]
        phase = Xc.angle()                 # [B, n_bins, T]
        Ylin = Tc.abs()                    # [B, n_bins, T]
        Yphase = Tc.angle()                # [B, n_bins, T]

        Xmel = torch.matmul(Xlin.transpose(1, 2), mel_basis.T)  # [B, T, M]
        Ymel = torch.matmul(Ylin.transpose(1, 2), mel_basis.T)  # [B, T, M]

        logmel = torch.log(Xmel + log_eps)
        x = (logmel - mel_mean) / mel_std  # [B, T, M]
        x = x.transpose(1, 2)              # [B, M, T]

        mask_lab = torch.clip(Ymel / (Xmel + mask_eps), 0.0, gmax)  # [B, T, M]
        mask_lab = mask_lab.transpose(1, 2)  # [B, M, T]

        Xmel = Xmel.transpose(1, 2)        # [B, M, T]
        Ymel = Ymel.transpose(1, 2)        # [B, M, T]
        Xlin = Xlin                        # [B, n_bins, T]
        Ylin = Ylin                        # [B, n_bins, T]

        mask, dphi, noise = model(x, inst_id)
        Lm = mask_l1(mask, mask_lab)
        Llog = logmel_l1(mask, Xmel, Ymel)
        loss = tc.w_mask * Lm + tc.w_logmel * Llog
        if tc.w_mrstft > 0:
            y = recon(Xlin, phase, mask, dphi, noise, length=win_samples)
            loss = loss + tc.w_mrstft * multi_res_stft(y, t_wav, tc.mrstft_ffts, tc.mrstft_hops)
        if tc.use_phase and tc.w_cplx > 0:
            loss = loss + tc.w_cplx * complex_stft_loss(
                Xlin, phase, mask, dphi, recon.mel_inv, recon.phase_inv,
                Ylin, Yphase)
        return loss, Llog

    for ep in range(tc.epochs):
        model.train()
        tl = 0.0
        n = 0
        for b in dl_tr:
            loss, _ = run_batch(b, True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tl += loss.item()
            n += 1
        msg = f"epoch {ep:3d} train_loss={tl / max(n, 1):.4f}"

        val_metric: Optional[float] = None
        if dl_va is not None:
            model.eval()
            vlog = 0.0
            nv = 0
            with torch.no_grad():
                for b in dl_va:
                    _, Llog = run_batch(b, False)
                    vlog += float(Llog)
                    nv += 1
            vlog /= max(nv, 1)
            val_metric = vlog
            msg += f"  val_logmel={vlog:.4f}"
            if vlog < best_val:
                best_val = vlog
                torch.save(_ckpt_payload(), ckpt_dir / "best.pt")
        print(f"[mask.train_gpu] {msg}")

        if on_epoch is not None:
            try:
                on_epoch(ep, float(tl / max(n, 1)), val_metric, float(best_val))
            except Exception:
                pass
        if should_stop is not None and should_stop():
            print("[mask.train_gpu] 收到停止请求，提前结束。")
            break

    torch.save(_ckpt_payload(), ckpt_dir / "last.pt")
    return {"ckpt_dir": str(ckpt_dir), "best_val_logmel": f"{best_val:.6f}"}


__all__ = ["train", "train_gpu"]
