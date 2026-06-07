"""训练条件 DDSP 模型：划分数据→标准化→训练→渲染 demo→导出产物。

按 sample_id 划分 train/val/test，避免同一源样本的不同目标泄漏到验证集。
训练中周期性渲染各目标乐器 demo 及“乐器切换”demo，便于主观验证。

用法：
  python scripts/train.py --run demo_run --epochs 60
"""

import _paths  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from esp_xform.audio.io import save_wav
from esp_xform.config import load_config, load_instruments
from esp_xform.data.dataset import DDSPFrameDataset, collate_sequences, compute_feature_stats
from esp_xform.ddsp.bands import make_band_edges
from esp_xform.infer import predict_params, synth_from_params, build_switch_schedule
from esp_xform.models import ConditionalDDSPNet
from esp_xform.train.export import export_artifacts
from esp_xform.train.trainer import Trainer


def split_by_sample(example_paths, val_ratio, test_ratio, seed):
    by_sample = {}
    for p in example_paths:
        sid = Path(p).stem.split("__")[0]
        by_sample.setdefault(sid, []).append(p)
    sids = sorted(by_sample)
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    n = len(sids)
    n_test = max(1, int(round(test_ratio * n))) if n > 2 else 0
    n_val = max(1, int(round(val_ratio * n))) if n > 1 else 0
    test_ids = set(sids[:n_test])
    val_ids = set(sids[n_test : n_test + n_val])
    splits = {"train": [], "val": [], "test": []}
    for sid in sids:
        key = "test" if sid in test_ids else "val" if sid in val_ids else "train"
        splits[key].extend(by_sample[sid])
    # 保证 train/val 非空
    if not splits["train"]:
        splits["train"] = splits["val"] or splits["test"]
    if not splits["val"]:
        splits["val"] = splits["train"]
    return splits


def make_render_cb(example_path, cfg, inst_cfg, mean, std, band_edges, device, render_dir):
    """返回训练期 demo 渲染回调：对固定样本渲染各目标乐器 + 切换 demo。"""
    data = np.load(example_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    f0 = data["f0_hz"].astype(np.float32)
    inst_ids = inst_cfg["instruments"]
    targets = inst_cfg["target_instruments"]
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    def cb(epoch, trainer):
        model = trainer.model
        for tgt in targets:
            iid = inst_ids[tgt]
            params = predict_params(model, features, iid, mean, std, device)
            audio = synth_from_params(params, f0, cfg, band_edges)
            save_wav(render_dir / f"epoch_{epoch:03d}_{tgt}.wav", audio, cfg.audio.sample_rate)
        # 乐器切换 demo
        target_id_list = [inst_ids[t] for t in targets]
        T = features.shape[0]
        sched = build_switch_schedule(T, target_id_list, cfg.audio.hop_size, cfg.audio.sample_rate, 1.5)
        params_by = {i: predict_params(model, features, i, mean, std, device) for i in set(target_id_list)}
        out = np.stack([params_by[int(sched[t])][t] for t in range(T)]).astype(np.float32)
        audio = synth_from_params(out, f0, cfg, band_edges)
        save_wav(render_dir / f"epoch_{epoch:03d}_switch.wav", audio, cfg.audio.sample_rate)
        print(f"  [render] epoch {epoch}: 已渲染 {len(targets)} 个乐器 + 切换 demo")

    return cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ddsp.yaml")
    ap.add_argument("--instruments", default="configs/instruments.yaml")
    ap.add_argument("--run", default="demo_run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stft", action="store_true", help="启用多尺度 STFT 损失")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inst_cfg = load_instruments(args.instruments)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.stft:
        cfg.train.use_stft_loss = True
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    label_dir = Path(cfg.paths.labels_dir)
    example_paths = sorted(str(p) for p in label_dir.glob("*.npz"))
    if not example_paths:
        raise SystemExit(f"{label_dir} 下没有训练样本，请先运行 extract_features.py / extract_labels.py")

    run_dir = Path(cfg.paths.output_dir) / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(cfg.paths.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    splits = split_by_sample(example_paths, cfg.train.val_ratio, cfg.train.test_ratio, cfg.train.seed)
    (splits_dir / "splits.json").write_text(
        json.dumps({k: [Path(p).name for p in v] for k, v in splits.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"数据划分：train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} (样本窗口前)")

    mean, std = compute_feature_stats(splits["train"], cfg.feature.feature_dim)

    train_ds = DDSPFrameDataset(splits["train"], cfg.train.sequence_length, mean, std)
    val_ds = DDSPFrameDataset(splits["val"], cfg.train.sequence_length, mean, std)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=collate_sequences, num_workers=cfg.train.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            collate_fn=collate_sequences, num_workers=cfg.train.num_workers)
    print(f"窗口数：train={len(train_ds)} val={len(val_ds)}  | device={device}")

    num_instruments = max(inst_cfg["instruments"].values()) + 1
    model = ConditionalDDSPNet(cfg, num_instruments)
    print(f"模型参数量：{model.num_parameters()}")

    band_edges = make_band_edges(cfg.audio.sample_rate, cfg.audio.fft_size, cfg.ddsp.n_noise_bands)
    trainer = Trainer(model, cfg, device, train_loader, val_loader, run_dir, band_edges,
                      feature_mean=mean, feature_std=std, instruments=inst_cfg)

    render_example = splits["val"][0] if splits["val"] else splits["train"][0]
    render_cb = make_render_cb(render_example, cfg, inst_cfg, mean, std, band_edges, device, run_dir / "renders")

    history = trainer.fit(render_cb=render_cb)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    paths = export_artifacts(model, cfg, mean, std, inst_cfg, run_dir,
                             extra={"run": args.run, "best_val_param_loss": trainer.best_val})
    print("导出产物：")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"\n训练完成。best_val_param_loss={trainer.best_val:.5f}")
    print(f"渲染 demo 见：{run_dir / 'renders'}")


if __name__ == "__main__":
    main()
