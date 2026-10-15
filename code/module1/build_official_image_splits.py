
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT.parent
    / "FAR-POLYP-SEG-Official"
    / "splits"
    / "far_internal_cv5"
)

MANIFEST = ROOT / "data" / "dataset_manifest.csv"
PATIENT_SPLITS = ROOT / "data" / "patient_splits_official_cv5.csv"
OUTPUT = ROOT / "data" / "official_polyp_image_splits_cv5.csv"

# 1. Đọc manifest và danh sách patient split
manifest = pd.read_csv(MANIFEST)
patient_splits = pd.read_csv(PATIENT_SPLITS)

# Baseline reproduce: lấy danh sách ảnh polyp
positive = manifest[manifest["is_polyp"] == 1].copy()

# Định dạng tên ảnh tương ứng JSON gốc của tác giả
positive["official_key"] = positive.apply(
    lambda r: (
        str(r["patient_id"])
        + "/"
        + Path(r["original_path"]).name
    ),
    axis=1
)

assert len(positive) == 432
assert positive["official_key"].is_unique

lookup = positive.set_index("official_key")

rows = []

# 2. Đọc chính xác danh sách ảnh trong 5 JSON
for fold_id in range(5):

    with open(
        OFFICIAL / f"fold{fold_id}.json",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    assert data["fold"] == fold_id
    assert data["seed"] == 2026

    for split in ["train", "val", "test"]:

        official_images = data[split]

        # Đối chiếu bệnh nhân với file CSV đã tạo
        csv_patients = set(
            patient_splits.loc[
                (patient_splits["fold"] == fold_id)
                & (
                    patient_splits["official_split"]
                    == split.upper()
                ),
                "patient_id"
            ]
        )

        actual_patients = {
            key.split("/")[0]
            for key in official_images
        }

        assert csv_patients == actual_patients

        for key in official_images:

            assert key in lookup.index, (
                f"Missing image: {key}"
            )

            sample = lookup.loc[key]

            rows.append({
                "fold": fold_id,
                "split": split.upper(),
                "image_id": sample["image_id"],
                "patient_id": sample["patient_id"],
                "original_path": sample["original_path"],
                "mask_path": sample["mask_path"],
                "is_polyp": 1,
                "official_key": key
            })

# 3. Xuất CSV
df = pd.DataFrame(rows)

assert len(df) == 432 * 5
assert not df.duplicated(
    ["fold", "image_id"]
).any()

# Kiểm tra mỗi fold có 432 ảnh polyp
assert (df.groupby("fold").size() == 432).all()

df = df.sort_values(
    ["fold", "split", "patient_id", "image_id"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

# 4. Báo cáo
print("=== OFFICIAL IMAGE SPLIT CHECK ===")

print(pd.crosstab(
    df["fold"],
    df["split"]
).to_string())

print()
print("Total rows:", len(df))
print("Unique images:", df["image_id"].nunique())
print("Images per fold:", df.groupby("fold").size().tolist())
print("Only polyp images:", bool((df["is_polyp"] == 1).all()))
print("Saved:", OUTPUT)
print("RESULT: PASS")
