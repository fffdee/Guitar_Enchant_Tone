"""验证：在数据加载器上计算参数级误差指标。"""

from __future__ import annotations

from typing import Dict

import torch

from ..ddsp.losses import param_loss


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> Dict[str, float]:
    """返回 {harmonic_mae, noise_mae, param_loss}（按样本数加权平均）。"""
    model.eval()
    nh = cfg.ddsp.n_harmonics
    w = cfg.train.noise_loss_weight
    tot = {"harmonic_mae": 0.0, "noise_mae": 0.0, "param_loss": 0.0}
    n = 0
    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)
        inst = batch["instrument_id"].to(device)
        pred = model(feats, inst)

        harm_mae = (pred[..., :nh] - labels[..., :nh]).abs().mean()
        noise_mae = (pred[..., nh:] - labels[..., nh:]).abs().mean()
        pl = param_loss(pred, labels, nh, w)

        bs = feats.size(0)
        tot["harmonic_mae"] += float(harm_mae) * bs
        tot["noise_mae"] += float(noise_mae) * bs
        tot["param_loss"] += float(pl) * bs
        n += bs

    if n == 0:
        return tot
    return {k: v / n for k, v in tot.items()}


__all__ = ["evaluate"]
