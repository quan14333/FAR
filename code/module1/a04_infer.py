import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from clean_v2_dataset import CleanV2SegDataset, load_fold_index

SPLIT_ORDER = ["SEG_TRAIN", "SEG_VAL", "META_FIT", "CALIBRATION"]
EXPECTED = {
    "SEG_TRAIN": 1318,
    "SEG_VAL": 356,
    "META_FIT": 610,
    "CALIBRATION": 433,
}
CKPT_SHA = (
    "5268013f6fd90d7c47412875b60b19ef00c37d9"
    "de367dd3524cda9368a0c1a61"
)
UPSTREAM_PIN = "ad3bf1c51529bda6de50badedba124e1e15a653c"
THRESHOLD = 0.5
INPUT_SIZE = 256


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--notebook-version", default="unrecorded")
    args = parser.parse_args()

    if args.mode == "full" and args.notebook_version == "unrecorded":
        raise ValueError("Record Kaggle notebook version for full inference")

    root = Path(__file__).resolve().parents[2]

    for p in [args.dataset, args.splits, args.checkpoint, args.upstream]:
        assert p.exists(), f"MISSING: {p}"

    assert git_head(args.upstream) == UPSTREAM_PIN
    assert digest(args.checkpoint) == CKPT_SHA
    print("CHECKPOINT SHA256: PASS")
    print("UPSTREAM COMMIT: PASS")

    sys.path.insert(0, str(args.upstream))
    from farseg.models.pranet import PraNet

    frame = load_fold_index(args.splits, fold=0)

    actual = (
        frame[frame.inner_split.isin(SPLIT_ORDER)]
        .groupby("inner_split").size().to_dict()
    )
    assert actual == EXPECTED
    assert sum(actual.values()) == 2717

    datasets = {}
    for split in SPLIT_ORDER:
        datasets[split] = CleanV2SegDataset(
            args.dataset, args.splits,
            fold=0, split=split, size=INPUT_SIZE,
            include_healthy=True,
            augment=False, return_metadata=True
        )
        assert len(datasets[split]) == EXPECTED[split]

    # Smoke: 1 Healthy + 1 Polyp per split,
    # plus 1 more of each category from SEG_VAL.
    jobs = []

    if args.mode == "smoke":
        for split in SPLIT_ORDER:
            ds = datasets[split]
            for category in ["Healthy", "Polyp"]:
                idxs = ds.rows.index[
                    ds.rows["category"] == category
                ].tolist()

                needed = 2 if split == "SEG_VAL" else 1
                assert len(idxs) >= needed

                for ix in idxs[:needed]:
                    jobs.append((split, int(ix)))

        assert len(jobs) == 10
    else:
        for split in SPLIT_ORDER:
            for ix in range(len(datasets[split])):
                jobs.append((split, ix))
        assert len(jobs) == 2717

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = PraNet(channel=32, pretrained=False)
    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    run_id = (
        f"a04_fold0_{args.mode}_{CKPT_SHA[:12]}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    out = args.out_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    metadata = {
        "status": "RUNNING",
        "run_id": run_id,
        "mode": args.mode,
        "fold": 0,
        "protocol": "inner_clean_v2_v1",
        "expected_images": len(jobs),
        "checkpoint_sha256": CKPT_SHA,
        "checkpoint_path": str(args.checkpoint),
        "source_repo_commit": git_head(root),
        "upstream_commit": UPSTREAM_PIN,
        "split_csv_sha256": digest(args.splits),
        "transform_sha256": digest(
            root / "code/module1/clean_v2_transforms.py"
        ),
        "inference_script_sha256": digest(Path(__file__)),
        "preprocessing": (
            "CleanV2SegDataset + JointSegTransform; "
            "RGB; PIL bilinear resize 256x256; "
            "ImageNet mean/std; augment=False"
        ),
        "head": "DeepSupervisionOutput.main",
        "prediction_type": "float32_native_sigmoid_probability",
        "resize": "logits_bilinear_align_corners_false",
        "threshold": THRESHOLD,
        "binary_values": [0, 255],
        "notebook_slug": os.getenv("KAGGLE_KERNEL_SLUG", "unavailable"),
        "notebook_version": args.notebook_version,
        "device": str(device),
        "torch_version": torch.__version__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = out / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    records = []

    with torch.inference_mode():
        for n, (split, ix) in enumerate(jobs, 1):
            ds = datasets[split]
            sample = ds[ix]
            row = ds.rows.iloc[ix]

            x = sample["image"].unsqueeze(0).to(device)
            assert tuple(x.shape) == (1, 3, 256, 256)

            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )

            with context:
                raw = model(x)

            assert hasattr(raw, "main")
            logits = raw.main.float()
            assert logits.ndim == 4 and logits.shape[1] == 1

            h = int(sample["native_height"])
            w = int(sample["native_width"])

            native_logits = F.interpolate(
                logits, size=(h, w),
                mode="bilinear", align_corners=False
            )
            prob = (
                native_logits.sigmoid()
                .squeeze(0).squeeze(0)
                .cpu().numpy()
                .astype(np.float32)
            )

            assert prob.shape == (h, w)
            assert prob.dtype == np.float32
            assert prob.size > 0
            assert np.isfinite(prob).all()
            assert np.min(prob) >= 0 and np.max(prob) <= 1

            mask = ((prob >= THRESHOLD).astype(np.uint8) * 255)

            image_id = str(row["image_id"])
            stem = hashlib.sha256(
                image_id.encode("utf-8")
            ).hexdigest()

            rel_npy = Path("probabilities") / split / f"{stem}.npy"
            rel_png = Path("masks") / split / f"{stem}.png"
            p_npy = out / rel_npy
            p_png = out / rel_png

            p_npy.parent.mkdir(parents=True, exist_ok=True)
            p_png.parent.mkdir(parents=True, exist_ok=True)

            assert not p_npy.exists() and not p_png.exists()

            np.save(p_npy, prob, allow_pickle=False)
            Image.fromarray(mask, mode="L").save(p_png)

            # Read saved outputs to verify file integrity.
            saved = np.load(p_npy, allow_pickle=False)
            with Image.open(p_png) as im:
                saved_mask = np.asarray(im.convert("L"))

            assert saved.shape == (h, w)
            assert saved.dtype == np.float32
            assert saved_mask.shape == (h, w)
            assert np.array_equal(saved_mask, mask)

            records.append({
                "run_id": run_id,
                "fold": 0,
                "split": split,
                "outer_split": str(row["outer_split"]),
                "image_id": image_id,
                "patient_id": str(row["patient_id"]),
                "split_group_id": str(row["split_group_id"]),
                "category": str(row["category"]),
                "source_image_path": str(row["image_path"]),
                "source_image_sha256": str(row["sha256"]),
                "source_mask_path": str(row["mask_path"]),
                "checkpoint_sha256": CKPT_SHA,
                "prediction_type": "sigmoid_probability",
                "input_size": INPUT_SIZE,
                "native_height": h,
                "native_width": w,
                "preprocessing_version": metadata["transform_sha256"],
                "threshold": THRESHOLD,
                "notebook_version": args.notebook_version,
                "probability_path": str(rel_npy),
                "binary_mask_path": str(rel_png),
                "probability_sha256": digest(p_npy),
                "binary_mask_sha256": digest(p_png),
                "prob_min": float(prob.min()),
                "prob_max": float(prob.max()),
                "positive_pixels": int((mask > 0).sum()),
            })

            if args.mode == "smoke" or n % 100 == 0:
                print(
                    f"[{n}/{len(jobs)}] {split} "
                    f"{sample['category']} "
                    f"{w}x{h} PASS",
                    flush=True
                )

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    assert len(records) == len(jobs)

    metadata["status"] = "COMPLETE"
    metadata["completed_images"] = len(records)
    metadata["manifest_sha256"] = digest(manifest)
    metadata["completed_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("OUTPUT:", out)
    print("MANIFEST:", manifest)
    print("STATUS: A04", args.mode.upper(), "PASS")


if __name__ == "__main__":
    main()
