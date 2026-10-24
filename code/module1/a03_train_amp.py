"""A03 reproducible AMP-only PraNet training (optional if old AMP can be proven).

IMPORTANT: This is a NEW run and checkpoint, not proof that a historical
checkpoint was trained with AMP. Re-run A04 inference if selecting this model.
Never loads META_FIT, CALIBRATION, OUTER_TEST, or NEVER_POLYP_TEST data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from clean_v2_dataset import CleanV2SegDataset, _source_path
from a03_verify import digest

DEFAULT_SPLIT = Path(__file__).resolve().parents[2] / "data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv"
UPSTREAM_COMMIT = "ad3bf1c51529bda6de50badedba124e1e15a653c"


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def structure_loss(logits, targets):
    """Standard PraNet weighted BCE plus weighted IoU loss."""
    import torch
    import torch.nn.functional as F
    weight = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
    wbce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    probability = logits.sigmoid()
    intersection = ((probability * targets) * weight).sum(dim=(2, 3))
    union = ((probability + targets) * weight).sum(dim=(2, 3))
    wiou = 1 - (intersection + 1) / (union - intersection + 1)
    return (wbce + wiou).mean()


def heads(prediction, target_size):
    """Unpack upstream PraNet DeepSupervisionOutput without detaching gradients.

    Upstream may return a dataclass / namedtuple / dict holding one primary
    logits tensor and auxiliary segmentation heads, rather than a tensor tuple.
    The main prediction MUST be last: evaluate() uses heads(...)[-1] for Dice.
    """
    import re
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass

    import torch
    import torch.nn.functional as F

    discovered = []
    seen = set()

    def visit(item, label="output", depth=0):
        if depth > 8:
            raise ValueError("PraNet output nesting too deep")
        if isinstance(item, torch.Tensor):
            discovered.append((label, item))
        elif isinstance(item, Mapping):
            for key, value in item.items():
                visit(value, f"{label}.{key}", depth + 1)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{label}.{field.name}", depth + 1)
        elif isinstance(item, tuple) and hasattr(item, "_fields"):
            for key in item._fields:
                visit(getattr(item, key), f"{label}.{key}", depth + 1)
        elif isinstance(item, (list, tuple)):
            for i, value in enumerate(item):
                visit(value, f"{label}[{i}]", depth + 1)
        elif hasattr(item, "__dict__"):
            if id(item) in seen:
                raise ValueError("Cyclic PraNet model output")
            seen.add(id(item))
            for key, value in vars(item).items():
                if not key.startswith("_"):
                    visit(value, f"{label}.{key}", depth + 1)
            seen.remove(id(item))
        # Ignore scalar metadata, if any.

    visit(prediction)
    if not discovered:
        raise TypeError(f"No tensor found in PraNet output {type(prediction).__name__}")
    for label, tensor in discovered:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(f"Unexpected PraNet head {label}: {tuple(tensor.shape)}")

    primary_keys = {"main", "main_logits", "logits", "final", "final_logits",
                    "prediction", "pred", "out", "output", "primary"}
    def is_primary(label):
        parts = [x for x in re.split(r"[.\[\]]+", label.lower()) if x]
        # The generic root name 'output' is not proof that a head is primary.
        return any(part in primary_keys for part in parts[1:])

    primary = [(label, tensor) for label, tensor in discovered if is_primary(label)]
    auxiliary = [(label, tensor) for label, tensor in discovered if not is_primary(label)]
    if len(discovered) > 1 and len(primary) != 1:
        if isinstance(prediction, (tuple, list)):
            # Historical tuple convention: final segmentation output last.
            ordered = discovered
        else:
            raise ValueError(
                "Ambiguous main PraNet head. Discovered paths: "
                + ", ".join(label for label, _ in discovered)
            )
    else:
        ordered = auxiliary + primary if primary else discovered
    result = []
    for _label, tensor in ordered:
        if tensor.shape[-2:] != tuple(target_size):
            tensor = F.interpolate(tensor, size=target_size, mode="bilinear",
                                   align_corners=False)
        result.append(tensor)
    return result


def evaluate(model, dataset, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    losses = []
    dices = []
    with torch.inference_mode():
        for ix in range(len(dataset)):
            inp, target = dataset[ix]
            x = inp.unsqueeze(0).to(device, non_blocking=True)
            target = target.unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                raw = model(x)
                layer = heads(raw, target.shape[-2:])
                loss = sum(structure_loss(v.float(), target) for v in layer)
            losses.append(float(loss.item()))
            row = dataset.rows.iloc[ix]
            with Image.open(_source_path(dataset.data_root, row.mask_path)) as m:
                native_gt = (np.asarray(m.convert("L"), dtype=np.uint8) > 127)
            native_logits = F.interpolate(layer[-1].float(), size=native_gt.shape,
                                          mode="bilinear", align_corners=False)
            native_pred = (native_logits.sigmoid().squeeze().cpu().numpy() >= 0.5)
            intersection = int(np.logical_and(native_pred, native_gt).sum())
            denom = int(native_pred.sum()) + int(native_gt.sum())
            dices.append((2.0 * intersection / denom) if denom else 0.0)
    return float(np.mean(losses)), float(np.mean(dices))


def save_curves(history: list[dict], out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = [r["epoch"] for r in history]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].plot(x, [r["train_loss"] for r in history], label="Train loss")
    axs[0].plot(x, [r["val_loss"] for r in history], label="Val loss")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[1].plot(x, [r["val_dice"] for r in history], label="Val native Dice")
    axs[1].legend()
    axs[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="AMP PraNet Fold 0 Clean V2 training")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available():
        raise RuntimeError("A03 requires CUDA for AMP. Enable Kaggle GPU P100/T4.")
    if not args.upstream.is_dir():
        raise FileNotFoundError(args.upstream)
    source_commit = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
        text=True).strip()
    if source_commit != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected FAR upstream commit {source_commit}; expected {UPSTREAM_COMMIT}")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Output already exists; refusing to overwrite checkpoint: {args.out}")
    if args.max_epochs < 2 or args.batch_size < 1:
        raise ValueError("Invalid max-epochs/batch-size")
    args.out.mkdir(parents=True)
    # Actual run-time configuration (unlike historical reconstructed config).
    config_path = args.out / "train_config.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        "config_status: recorded_at_start_of_training\n"
        "protocol: inner_clean_v2_v1\n"
        "fold: 0\n"
        f"seed: {args.seed}\n"
        f"batch_size: {args.batch_size}\n"
        "train_size: 256\n"
        f"max_epochs: {args.max_epochs}\n"
        f"learning_rate: {args.lr}\n"
        f"min_learning_rate: {args.min_lr}\n"
        f"early_stop_patience: {args.patience}\n"
        "training_split: SEG_TRAIN\n"
        "selection_split: SEG_VAL\n"
        "loss: sum_structure_loss_all_pranet_heads\n"
        "checkpoint_selection: max_native_val_dice\n"
        "amp_enabled: true\n"
        "amp_autocast: cuda_float16\n"
        "grad_scaler_enabled: true\n",
        encoding="utf-8")
    seed_everything(args.seed)
    sys.path.insert(0, str(args.upstream.resolve()))
    from farseg.models.pranet import PraNet
    train = CleanV2SegDataset(args.dataset, args.splits, 0, "SEG_TRAIN", augment=True, size=256)
    val = CleanV2SegDataset(args.dataset, args.splits, 0, "SEG_VAL", augment=False, size=256)
    if len(train) != 215 or len(val) != 44:
        raise ValueError("Approved SEG_TRAIN/SEG_VAL Polyp counts changed")
    tr_groups = set(train.rows.split_group_id)
    va_groups = set(val.rows.split_group_id)
    if tr_groups & va_groups:
        raise AssertionError("Training group appears in validation")
    train.rows.to_csv(args.out / "train_images.csv", index=False)
    val.rows.to_csv(args.out / "val_images.csv", index=False)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("Starting fresh PraNet with ImageNet initialization; NO legacy model checkpoint is used")
    model = PraNet(channel=32, pretrained=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if not scaler.is_enabled():
        raise RuntimeError("CUDA GradScaler not enabled")
    columns = ["epoch", "train_loss", "val_loss", "val_dice", "learning_rate",
               "amp_enabled", "grad_scaler_enabled"]
    history = []
    best = -1.0
    best_epoch = 0
    stopped = 0
    started = time.monotonic()
    best_path = args.out / "best_model.pth"
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for xs, ys in loader:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = heads(model(xs), ys.shape[-2:])
                loss = sum(structure_loss(head.float(), ys) for head in logits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite train loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_loss, val_dice = evaluate(model, val, "cuda")
        lr_used = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": val_loss, "val_dice": val_dice, "learning_rate": lr_used,
               "amp_enabled": True, "grad_scaler_enabled": bool(scaler.is_enabled())}
        history.append(row)
        with (args.out / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(history)
        improved = val_dice > best + 1e-10
        if improved:
            best = val_dice
            best_epoch = epoch
            stopped = 0
            torch.save(model.state_dict(), best_path)
        else:
            stopped += 1
        print(f"Epoch {epoch:3d} | train_loss={row['train_loss']:.4f} | "
              f"val_loss={val_loss:.4f} | val_dice={val_dice:.5f} | "
              f"best={best:.5f}@{best_epoch} | AMP={scaler.is_enabled()}", flush=True)
        scheduler.step()
        if stopped >= args.patience:
            print(f"Early stop at {epoch}")
            break
    # Strict reload before reporting training complete.
    check_model = PraNet(channel=32, pretrained=False).cpu()
    state = torch.load(best_path, map_location="cpu", weights_only=True)
    check_model.load_state_dict(state, strict=True)
    del check_model
    cfg = {
        "model": "PraNet", "model_source": "FAR-POLYP-SEG/Codes",
        "model_source_commit": source_commit,
        "fold": 0, "seed": args.seed,
        "split_version": "inner_clean_v2_v1", "split_csv_sha256": digest(args.splits),
        "train_split": "SEG_TRAIN", "val_split": "SEG_VAL",
        "train_polyp_images": len(train), "val_polyp_images": len(val),
        "training_groups": sorted(tr_groups),
        "old_fold0_checkpoint_loaded": False,
        "pretrained": "ImageNet ResNet50 initialization",
        "input_size": 256, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "learning_rate": args.lr,
        "min_learning_rate": args.min_lr, "early_stop_patience": args.patience,
        "mask_threshold": 127, "prediction_threshold": 0.5,
        "checkpoint_selection": "best mean native-resolution val Dice",
        "amp_enabled": True, "grad_scaler_enabled": True,
        "amp_evidence": "torch.autocast cuda float16 + torch.amp.GradScaler cuda in every train epoch",
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "train_config_sha256": digest(config_path),
        "status": "TRAINING_COMPLETE", "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_val_dice": best,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": digest(best_path),
        "history_csv": str(args.out / "history.csv"),
        "train_seconds_this_session": round(time.monotonic() - started, 2),
    }
    (args.out / "run_metadata.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    save_curves(history, args.out / "training_curves.png")
    print("STATUS: A03 TRAINING COMPLETE, AMP ENABLED")
    print("Note: if this checkpoint replaces the previous one, rerun A04 inference with THIS exact SHA256.")


if __name__ == "__main__":
    main()
