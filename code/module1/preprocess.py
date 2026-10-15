
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["FAR_DATA_ROOT"])
MANIFEST = ROOT / "data" / "dataset_manifest.csv"

IMAGE_SIZE = (256, 256)

def load_sample(row):
    # Đọc ảnh RGB
    image_path = DATA_ROOT / row["original_path"]

    with Image.open(image_path) as img:
        image = img.convert("RGB").copy()

    # Đọc ground-truth mask
    if int(row["is_polyp"]) == 1:
        mask_path = DATA_ROOT / row["mask_path"]

        with Image.open(mask_path) as m:
            gray = np.asarray(m.convert("L"))

        # JPEG mask -> binary (0 hoặc 255)
        binary = (gray >= 128).astype(np.uint8) * 255

    else:
        # Ảnh Healthy: tạo mask toàn nền
        binary = np.zeros(
            (image.height, image.width),
            dtype=np.uint8
        )

    assert binary.shape == (image.height, image.width)

    # Resize ảnh bằng bilinear
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # Resize mask bằng nearest neighbor
    mask = Image.fromarray(binary).resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    # Chuyển sang NumPy
    image = np.asarray(image, dtype=np.float32) / 255.0
    mask = (np.asarray(mask) >= 128).astype(np.float32)

    return image, mask


if __name__ == "__main__":
    df = pd.read_csv(MANIFEST)

    # Thử với một ảnh có polyp
    row = df[df["is_polyp"] == 1].iloc[0]

    image, mask = load_sample(row)

    print("=== PREPROCESS TEST ===")
    print("Image ID:", row["image_id"])
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Mask shape:", mask.shape)
    print("Mask unique:", np.unique(mask))
    print("Image range:", image.min(), image.max())
    print("Mask foreground:", int(mask.sum()))

    assert image.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)).issubset({0.0, 1.0})

    print("RESULT: PASS")
