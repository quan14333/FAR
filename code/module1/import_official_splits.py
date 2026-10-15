
import json
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT.parent
    / "FAR-POLYP-SEG-Official"
    / "splits"
    / "far_internal_cv5"
)

COHORT_FILE = ROOT / "data" / "patient_cohorts.csv"
OUTPUT = ROOT / "data" / "patient_splits_official_cv5.csv"

cohorts = pd.read_csv(COHORT_FILE)

assert len(cohorts) == 455
assert cohorts["patient_id"].is_unique

polyp_ids = set(
    cohorts.loc[
        cohorts["cohort"] == "POLYP_COHORT",
        "patient_id"
    ]
)

never_ids = set(
    cohorts.loc[
        cohorts["cohort"] == "NEVER_POLYP_COHORT",
        "patient_id"
    ]
)

assert len(polyp_ids) == 187
assert len(never_ids) == 268
assert polyp_ids.isdisjoint(never_ids)

rows = []
test_counter = Counter()

for fold_id in range(5):

    path = OFFICIAL / f"fold{fold_id}.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["fold"] == fold_id
    assert data["scheme"] == "grouped_cv5"
    assert data["seed"] == 2026

    groups = {
        split: set(data[f"{split}_patients"])
        for split in ["train", "val", "test"]
    }

    # Không có patient leakage
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

    assert set().union(*groups.values()) == polyp_ids

    for split in ["train", "val", "test"]:
        for patient_id in sorted(groups[split]):
            rows.append({
                "fold": fold_id,
                "patient_id": patient_id,
                "cohort": "POLYP_COHORT",
                "official_split": split.upper()
            })

    test_counter.update(groups["test"])

    # Never-polyp được giữ riêng ở mọi fold
    for patient_id in sorted(never_ids):
        rows.append({
            "fold": fold_id,
            "patient_id": patient_id,
            "cohort": "NEVER_POLYP_COHORT",
            "official_split": "NEVER_POLYP_TEST"
        })

# Kiểm tra mỗi patient được test đúng 1 lần
assert set(test_counter) == polyp_ids
assert all(n == 1 for n in test_counter.values())

df = pd.DataFrame(rows)

# Mỗi fold có đủ 455 bệnh nhân
assert len(df) == 455 * 5

assert not df.duplicated(
    ["fold", "patient_id"]
).any()

assert (df.groupby("fold").size() == 455).all()

df = df.sort_values(
    ["fold", "patient_id"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print("=== IMPORT OFFICIAL CV5 SPLITS ===")
print("Source: FAR-POLYP-SEG/Codes")
print("Official seed: 2026")
print()

print(pd.crosstab(
    df["fold"],
    df["official_split"]
).to_string())

print()
print("Total rows:", len(df))
print("Unique patients:", df["patient_id"].nunique())
print("Patients tested exactly once:", len(test_counter))
print("Saved:", OUTPUT)
print("RESULT: PASS")
