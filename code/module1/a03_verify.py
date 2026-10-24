"""A03 evidence audit for PraNet Clean V2 Fold 0, fail-closed for AMP.

Accepts a reconstructed historical metadata record and actual checkpoint/logs.
Reports PENDING when AMP is not evidenced; never backfills historical claims.
Optional strict architecture reload uses the pinned FAR-POLYP-SEG/Codes package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "results/module1/inner_clean_v2_v1/fold0"
DEFAULT_SPLITS = REPO / "data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv"
EXPECTED = {"SEG_TRAIN": 215, "SEG_VAL": 44}
KNOWN_CHECKPOINT_SHA = "8c75a4fb5675d0dc7985704dcbf2a44e2545f603645dec31ff38c61e9f91561a"


def digest(path: Path) -> str:
    obj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            obj.update(chunk)
    return obj.hexdigest()


def read_history(path: Path) -> tuple[list[dict[str, str]], dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError("History CSV is empty")
    fields = list(history[0])
    loss_cols = [k for k in fields if k.lower() in ("train_loss", "loss", "training_loss")]
    val_cols = [k for k in fields if k.lower() in ("val_loss", "validation_loss")]
    dice_cols = [k for k in fields if k.lower() in ("val_dice", "val_dice_native", "best_val_dice")]
    epochs = [int(float(row["epoch"])) for row in history] if "epoch" in fields else []
    if epochs and (len(set(epochs)) != len(epochs) or epochs != sorted(epochs)):
        raise ValueError("History epochs not unique / ordered")
    for key in loss_cols + val_cols + dice_cols:
        if any(not math.isfinite(float(row[key])) for row in history):
            raise ValueError(f"Nonfinite history column: {key}")
    return history, {"rows": len(history), "fields": fields,
                     "train_loss_logged": bool(loss_cols),
                     "validation_loss_logged": bool(val_cols),
                     "validation_dice_logged": bool(dice_cols),
                     "epochs": epochs}


def split_audit(split_csv: Path, metadata: dict) -> dict:
    # All 5 folds are loaded to verify the shared protocol source; no dataset images needed.
    import pandas as pd
    df = pd.read_csv(split_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = {"fold", "patient_id", "split_group_id", "image_id", "category", "inner_split", "outer_split"}
    if not need <= set(df.columns):
        raise ValueError(f"Missing split columns: {sorted(need-set(df.columns))}")
    if len(df) != 40365 or set(df.fold) != set("01234"):
        raise ValueError("Unexpected approved CV5 split size/folds")
    fold = df[df.fold == "0"].copy()
    if (len(fold), fold.patient_id.nunique(), fold.split_group_id.nunique()) != (8073, 455, 453):
        raise ValueError("Fold 0 counts differ from Clean V2")
    if fold.image_id.duplicated().any():
        raise ValueError("Duplicate image IDs")
    if fold.groupby("patient_id").inner_split.nunique().max() != 1:
        raise ValueError("Patient leakage")
    if fold.groupby("split_group_id").inner_split.nunique().max() != 1:
        raise ValueError("Group leakage")
    split_counts = {}
    groups = {}
    for split, expected in EXPECTED.items():
        subset = fold[(fold.inner_split == split) & (fold.category == "Polyp")]
        if len(subset) != expected:
            raise ValueError(f"{split} Polyp {len(subset)} != {expected}")
        split_counts[split] = len(subset)
        groups[split] = set(subset.split_group_id)
    if groups["SEG_TRAIN"] & groups["SEG_VAL"]:
        raise ValueError("Train/val linked-group overlap")
    for split in ("META_FIT", "CALIBRATION", "OUTER_TEST", "NEVER_POLYP_TEST"):
        g = set(fold.loc[fold.inner_split == split, "split_group_id"])
        if g & groups["SEG_TRAIN"]:
            raise ValueError(f"Train overlaps held-out {split}")
    csv_sha = digest(split_csv)
    expected_sha = metadata.get("split_csv_sha256")
    # Git on Windows can check out CSV with CRLF while the Kaggle training
    # manifest was LF. The byte-normalized digest must match exactly; do NOT
    # weaken this into a rows/counts-only checksum acceptance.
    normalized_sha = hashlib.sha256(split_csv.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if expected_sha not in (csv_sha, normalized_sha):
        raise ValueError("Split checksum does not match training metadata")
    return {"image_csv_sha256": csv_sha, "image_csv_normalized_lf_sha256": normalized_sha,
            "newline_normalization_used": csv_sha != expected_sha,
            "fold_images": len(fold),
            "fold_patients": fold.patient_id.nunique(),
            "fold_groups": fold.split_group_id.nunique(),
            "train_polyp": 215, "validation_polyp": 44,
            "patient_leakage": 0, "linked_group_leakage": 0}


def strict_checkpoint_reload(path: Path, upstream: Path) -> dict:
    import torch
    if not upstream.is_dir():
        raise FileNotFoundError("Upstream FAR-POLYP-SEG/Codes directory missing")
    sys.path.insert(0, str(upstream.resolve()))
    from farseg.models.pranet import PraNet
    # Avoid an unrelated ImageNet download for verification.
    model = PraNet(channel=32, pretrained=False)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Legacy torch with no weights_only kwarg.
        raise RuntimeError("Use a modern torch supporting weights_only=True for checkpoint audit")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint not a PyTorch state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return {"strict_load": True, "state_dict_tensors": len(state),
            "model": "farseg.models.pranet.PraNet(channel=32,pretrained=False)"}


def audit(*, run: Path, splits: Path, checkpoint: Path | None,
          history_path: Path, upstream: Path | None) -> dict:
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("fold"), metadata.get("split_version"),
        metadata.get("train_split"), metadata.get("val_split")) != (
            0, "inner_clean_v2_v1", "SEG_TRAIN", "SEG_VAL"):
        raise ValueError("Incorrect model/run/splits provenance")
    if metadata.get("old_fold0_checkpoint_loaded") is not False:
        raise ValueError("Legacy checkpoint initialization not ruled out")
    if (metadata.get("train_polyp_images"), metadata.get("val_polyp_images")) != (215, 44):
        raise ValueError("Training sample counts incorrect")
    split_result = split_audit(splits, metadata)
    result = {"task": "A03", "split": split_result,
              "run_metadata": {k: metadata.get(k) for k in (
                  "model", "seed", "epochs_completed", "best_epoch", "best_val_dice",
                  "batch_size", "learning_rate", "min_learning_rate", "early_stop_patience")}}
    missing = []
    if history_path.is_file():
        records, hist = read_history(history_path)
        result["history"] = hist
        if not hist["train_loss_logged"]:
            missing.append("train_loss_per_epoch")
        if not hist["validation_loss_logged"]:
            missing.append("val_loss_per_epoch")
        if not hist["validation_dice_logged"]:
            missing.append("val_dice_per_epoch")
        if hist["epochs"] and len(hist["epochs"]) != int(metadata["epochs_completed"]):
            missing.append("epoch_history_count_disagrees_with_run_metadata")
        if hist["validation_dice_logged"] and hist["epochs"]:
            key = next(k for k in records[0] if k.lower() in ("val_dice", "val_dice_native", "best_val_dice"))
            best = max(range(len(records)), key=lambda ix: float(records[ix][key]))
            if (hist["epochs"][best] != int(metadata["best_epoch"]) or
                abs(float(records[best][key]) - float(metadata["best_val_dice"])) > 1e-4):
                missing.append("best_epoch_or_dice_disagree_with_history")
        amp_series = "amp_enabled" in hist["fields"] and all(
            r["amp_enabled"].strip().lower() in ("true", "1") for r in records)
        scaler_series = "grad_scaler_enabled" in hist["fields"] and all(
            r["grad_scaler_enabled"].strip().lower() in ("true", "1") for r in records)
    else:
        result["history"] = {"available": False, "path": str(history_path)}
        missing.append("history_csv")
        amp_series = scaler_series = False
    # Never infer historical AMP from GPU presence, torch version, or checkpoint dtype.
    amp = metadata.get("amp_enabled") is True and amp_series and scaler_series
    result["amp"] = {"required": True, "verified": amp,
                      "evidence_rule": "run_metadata.amp_enabled=true AND all epochs in history report amp_enabled=true, grad_scaler_enabled=true"}
    if not amp:
        missing.append("executed_amp_training_provenance")
    expected = metadata.get("best_checkpoint_sha256")
    result["checkpoint"] = {"expected_sha256": expected, "available": bool(checkpoint and checkpoint.is_file())}
    if checkpoint is None or not checkpoint.is_file():
        missing.append("actual_checkpoint_file")
    else:
        observed = digest(checkpoint)
        result["checkpoint"]["actual_sha256"] = observed
        if observed != expected:
            raise ValueError("Checkpoint SHA256 mismatch with training metadata")
        if upstream is None:
            missing.append("upstream_path_to_strict_reload_checkpoint")
        else:
            result["checkpoint"].update(strict_checkpoint_reload(checkpoint, upstream))
    result["missing_evidence"] = missing
    result["status"] = "PASS" if not missing else "PENDING"
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="A03 acceptance, no training, no test-set access")
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--upstream", type=Path, help="FAR-POLYP-SEG/Codes clone containing farseg")
    p.add_argument("--history", type=Path, help="Actual history.csv from corresponding Kaggle run")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    history = args.history or args.run / "history.csv"
    try:
        report = audit(run=args.run, splits=args.splits, checkpoint=args.checkpoint,
                       history_path=history, upstream=args.upstream)
    except Exception as e:
        report = {"task": "A03", "status": "FAIL", "error": f"{type(e).__name__}: {e}"}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "a03_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("A03 status:", report["status"])
    if report["status"] != "PASS":
        print("Missing/issues:", report.get("missing_evidence", report.get("error")))
    print("Evidence:", target)
    return 0 if report["status"] == "PASS" else (2 if report["status"] == "PENDING" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
