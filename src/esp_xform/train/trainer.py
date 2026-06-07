"""训练器：参数 MAE（默认）+ 可选多尺度 STFT 损失。

STFT 损失采用“标签域一致”的代理：对预测参数与目标标签分别用 torch 合成器渲染
音频后比较多尺度 STFT，从而在不缓存原始波形的前提下提供音频域梯度，联合约束
谐波与噪声。默认关闭（cfg.train.use_stft_loss=False）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
import sys

import numpy as np
import torch
from torch.optim import AdamW

try:
    # 作为包模块导入（推荐路径）
    from ..ddsp.losses import param_loss, multi_scale_stft_loss
    from ..ddsp.synthesizer import ddsp_synth_torch
    from .validate import evaluate
except ImportError:
    # 兼容“直接运行本文件”的场景（__package__ 为空导致相对导入失败）
    # 文件位置: src/esp_xform/train/trainer.py -> src 在 parents[2]
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from esp_xform.ddsp.losses import param_loss, multi_scale_stft_loss  # type: ignore
    from esp_xform.ddsp.synthesizer import ddsp_synth_torch  # type: ignore
    from esp_xform.train.validate import evaluate  # type: ignore

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        device,
        train_loader,
        val_loader,
        output_dir: str | Path,
        band_edges: np.ndarray,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        instruments: Optional[Dict] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.band_edges = np.asarray(band_edges)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.instruments = instruments or {}

        self.nh = cfg.ddsp.n_harmonics
        self.noise_w = cfg.train.noise_loss_weight
        self.use_stft = bool(cfg.train.use_stft_loss)
        self.stft_w = cfg.train.stft_loss_weight
        self.grad_clip = cfg.train.grad_clip

        self.optimizer = AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.history: List[Dict] = []
        self.best_val = float("inf")

    # ------------------------------------------------------------------ #
    def _stft_loss(self, pred, labels, f0):
        a = self.cfg.audio
        harm_p, noise_p = pred[..., : self.nh], pred[..., self.nh :]
        harm_t, noise_t = labels[..., : self.nh], labels[..., self.nh :]
        harm_p = torch.clamp(harm_p, min=0.0)
        noise_p = torch.clamp(noise_p, min=0.0)
        y_pred = ddsp_synth_torch(
            f0, harm_p, noise_p, a.sample_rate, a.hop_size, a.fft_size, self.band_edges
        )
        y_tgt = ddsp_synth_torch(
            f0, harm_t, noise_t, a.sample_rate, a.hop_size, a.fft_size, self.band_edges
        )
        return multi_scale_stft_loss(y_pred, y_tgt)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = 0.0
        running_p = 0.0
        running_s = 0.0
        n = 0
        it = self.train_loader
        if _HAS_TQDM:
            it = tqdm(it, desc=f"epoch {epoch}", leave=False)

        for batch in it:
            feats = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            inst = batch["instrument_id"].to(self.device)
            f0 = batch["f0_hz"].to(self.device)

            pred = self.model(feats, inst)
            p_loss = param_loss(pred, labels, self.nh, self.noise_w)
            loss = p_loss
            s_loss_val = 0.0
            if self.use_stft:
                s_loss = self._stft_loss(pred, labels, f0)
                loss = loss + self.stft_w * s_loss
                s_loss_val = s_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bs = feats.size(0)
            running += loss.item() * bs
            running_p += p_loss.item() * bs
            running_s += s_loss_val * bs
            n += bs

        n = max(n, 1)
        return {
            "loss": running / n,
            "param_loss": running_p / n,
            "stft_loss": running_s / n,
        }

    def fit(
        self,
        epochs: Optional[int] = None,
        render_cb: Optional[Callable[[int, "Trainer"], None]] = None,
    ) -> List[Dict]:
        epochs = epochs or self.cfg.train.epochs
        render_every = self.cfg.train.render_every
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(epoch)
            val = evaluate(self.model, self.val_loader, self.cfg, self.device)
            dt = time.time() - t0

            rec = {"epoch": epoch, "time_s": round(dt, 2), **{f"train_{k}": round(v, 6) for k, v in tr.items()},
                   **{f"val_{k}": round(v, 6) for k, v in val.items()}}
            self.history.append(rec)
            print(
                f"[{epoch:03d}/{epochs}] "
                f"train_loss={tr['loss']:.5f} param={tr['param_loss']:.5f} "
                f"stft={tr['stft_loss']:.5f} | "
                f"val_param={val['param_loss']:.5f} "
                f"harm_mae={val['harmonic_mae']:.5f} noise_mae={val['noise_mae']:.5f} "
                f"({dt:.1f}s)"
            )

            if val["param_loss"] < self.best_val:
                self.best_val = val["param_loss"]
                self.save_checkpoint("best.pt", epoch, val)

            if render_cb and (epoch % render_every == 0 or epoch == epochs):
                try:
                    render_cb(epoch, self)
                except Exception as e:  # 渲染失败不应中断训练
                    print(f"  [render] 跳过：{e}")

        self.save_checkpoint("last.pt", epochs, val)
        return self.history

    def save_checkpoint(self, name: str, epoch: int, metrics: Dict) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "num_instruments": self.model.num_instruments,
                "epoch": epoch,
                "metrics": metrics,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "instruments": self.instruments,
            },
            self.ckpt_dir / name,
        )


__all__ = ["Trainer"]
