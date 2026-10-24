"""Patient-group-safe dataset for approved FAR Clean V2 / Inner CV5 splits.

No splitting/reassignment happens here. The only authority is the approved
``image_inner_splits_cv5_candidate.csv`` already published in PR #7.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    from .clean_v2_transforms import JointSegTransform
except ImportError:  # Standalone ``python code/module1/a02_verify.py``
    from clean_v2_transforms import JointSegTransform

SPLITS = (
    "SEG_TRAIN", "SEG_VAL", "META_FIT", "CALIBRATION",
    "OUTER_TEST", "NEVER_POLYP_TEST",
)
EXPECTED_OUTER = {
    "SEG_TRAIN": "TRAIN",
    "META_FIT": "TRAIN",
    "CALIBRATION": "TRAIN",
    "SEG_VAL": "VAL",
    "OUTER_TEST": "TEST",
    "NEVER_POLYP_TEST": "NEVER_POLYP_TEST",
}
COLUMNS = (
    "fold", "image_id", "patient_id", "split_group_id", "category",
    "image_path", "mask_path", "sha256", "outer_split", "inner_split",
)


def load_fold_index(csv_path: Union[str, Path], fold: Union[int, str] = 0,
                    *, strict_clean_v2: bool = True) -> pd.DataFrame:
    """Read the fixed approved split; validate *all* groups before filtering."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = set(COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV fields: {sorted(missing)}")
    rows = frame[frame["fold"] == str(fold)].copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No records for fold={fold}")
    if "protocol_version" not in rows.columns or not rows["protocol_version"].eq("inner_clean_v2_v1").all():
        raise ValueError("Unexpected protocol_version: approved Inner Clean V2 required")
    if rows["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id inside the selected fold")
    if not set(rows["inner_split"]).issubset(set(SPLITS)):
        raise ValueError("Unknown inner_split")
    if not set(rows["category"]).issubset({"Polyp", "Healthy"}):
        raise ValueError("Unknown category")
    if (rows.groupby("patient_id")["inner_split"].nunique() > 1).any():
        raise ValueError("Patient leakage across inner splits")
    if (rows.groupby("split_group_id")["inner_split"].nunique() > 1).any():
        raise ValueError("Linked patient group leakage across inner splits")
    if (rows.groupby("patient_id")["split_group_id"].nunique() > 1).any():
        raise ValueError("One patient has multiple linked group IDs")
    for inner, outer in EXPECTED_OUTER.items():
        group = rows[rows["inner_split"] == inner]
        if not group.empty and not (group["outer_split"] == outer).all():
            raise ValueError(f"Outer split mismatch for {inner}; expected {outer}")
    known_hashes = rows.loc[rows["sha256"].ne(""), ["sha256", "image_id"]]
    if known_hashes["sha256"].duplicated().any():
        raise ValueError("Repeated SHA-256 detected across distinct images")
    polyp = rows[rows["category"] == "Polyp"]
    if polyp["mask_path"].eq("").any():
        raise ValueError("Polyp sample without GT mask path")
    if rows.loc[rows["category"] == "Healthy", "mask_path"].ne("").any():
        raise ValueError("Unexpected GT mask for Healthy sample")
    if strict_clean_v2:
        numbers = (len(rows), rows["image_id"].nunique(),
                   rows["patient_id"].nunique(), rows["split_group_id"].nunique(),
                   len(polyp), int((rows["category"] == "Healthy").sum()))
        if numbers != (8073, 8073, 455, 453, 430, 7643):
            raise ValueError(f"Fold stats differ from approved Clean V2: {numbers}")
    return rows


def _source_path(root: Path, raw: str) -> Path:
    """Require a relative path inside the clean Dataset root."""
    if not raw or "\\" in raw:
        raise ValueError(f"Invalid/missing relative data path: {raw!r}")
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe data path: {raw}")
    if rel.parts and rel.parts[0] == "Dataset" and root.name == "Dataset":
        rel = Path(*rel.parts[1:])
    resolved_root = root.resolve()
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Data path escapes dataset root: {raw}")
    return resolved


class CleanV2SegDataset(Dataset):
    """Polyp GT + optional Healthy (zero GT) from ONE approved inner split.

    By default return ``(rgb, gt)`` as tensors shaped [3,S,S], [1,S,S].
    Pass ``return_metadata=True`` for dict with patient and image provenance.
    ``augment=None`` means enabled only on SEG_TRAIN. It is *forbidden* on
    validation, calibration, and test partitions.
    """
    def __init__(self, data_root: Union[str, Path], csv_path: Union[str, Path],
                 fold: Union[int, str], split: str, *, size: int = 256,
                 include_healthy: bool = False, augment: Optional[bool] = None,
                 return_metadata: bool = False,
                 strict_clean_v2: bool = True,
                 allow_final_test: bool = False) -> None:
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r}")
        if split in {"OUTER_TEST", "NEVER_POLYP_TEST"} and not allow_final_test:
            raise PermissionError(
                f"{split} is held out; set allow_final_test=True only for locked final evaluation"
            )
        if augment is None:
            augment = split == "SEG_TRAIN"
        if augment and split != "SEG_TRAIN":
            raise ValueError("Augmentation is forbidden outside SEG_TRAIN")
        self.data_root = Path(data_root)
        self.fold = str(fold)
        self.split = split
        self.return_metadata = return_metadata
        index = load_fold_index(csv_path, fold, strict_clean_v2=strict_clean_v2)
        index = index[index["inner_split"] == split]
        if not include_healthy:
            index = index[index["category"] == "Polyp"]
        self.rows = index.sort_values("image_id").reset_index(drop=True)
        self.transform = JointSegTransform(size=size, augment=augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image_path = _source_path(self.data_root, row["image_path"])
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
        if row["category"] == "Polyp":
            mask_path = _source_path(self.data_root, row["mask_path"])
            with Image.open(mask_path) as im:
                mask = im.convert("L")
        else:
            mask = Image.new("L", rgb.size, 0)
        native_w, native_h = rgb.size
        image_tensor, gt_tensor = self.transform(rgb, mask)
        if self.return_metadata:
            return {
                "image": image_tensor, "mask": gt_tensor,
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "split_group_id": row["split_group_id"],
                "fold": self.fold, "split": self.split,
                "category": row["category"],
                "native_height": native_h, "native_width": native_w,
            }
        return image_tensor, gt_tensor


def make_loader(data_root: Union[str, Path], csv_path: Union[str, Path],
                fold: Union[int, str], split: str, *, batch_size: int = 8,
                num_workers: int = 0, **kwargs) -> DataLoader:
    """Prevent shuffled evaluation/test loading (sampling is not re-splitting)."""
    dataset = CleanV2SegDataset(data_root, csv_path, fold, split, **kwargs)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(split == "SEG_TRAIN"),
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
