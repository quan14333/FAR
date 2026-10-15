
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

from preprocess import load_sample, MANIFEST

# Lấy 10 ảnh polyp từ 10 bệnh nhân khác nhau
df = pd.read_csv(MANIFEST)

samples = (
    df[df["is_polyp"] == 1]
    .sort_values("image_id")
    .drop_duplicates("patient_id")
    .sample(n=10, random_state=42)
)

SIZE = 352
ROW_HEIGHT = 380

# Xuất ảnh kiểm tra ngoài repository GitHub
output_dir = Path.home() / "Downloads" / "FAR_A02_preview"
output_dir.mkdir(parents=True, exist_ok=True)

canvas = Image.new(
    "RGB",
    (SIZE * 3, ROW_HEIGHT * 10),
    "white"
)

draw = ImageDraw.Draw(canvas)

print("=== VERIFY 10 IMAGE-MASK PAIRS ===")

for i, (_, row) in enumerate(samples.iterrows()):

    image, mask = load_sample(row)

    assert image.shape == (352, 352, 3)
    assert mask.shape == (352, 352)
    assert mask.sum() > 0

    # Ảnh RGB
    rgb = (image * 255).astype(np.uint8)

    # Mask trắng đen
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_rgb = np.stack([mask_uint8] * 3, axis=-1)

    # Overlay: vùng polyp màu đỏ
    overlay = rgb.astype(np.float32).copy()
    region = mask > 0.5

    overlay[region] = (
        0.5 * overlay[region]
        + 0.5 * np.array([255, 0, 0])
    )

    overlay = overlay.astype(np.uint8)

    y = i * ROW_HEIGHT

    draw.text(
        (5, y + 5),
        f"{i+1}. {row['patient_id']} | RGB / MASK / OVERLAY",
        fill="black"
    )

    canvas.paste(Image.fromarray(rgb), (0, y + 25))
    canvas.paste(Image.fromarray(mask_rgb), (SIZE, y + 25))
    canvas.paste(Image.fromarray(overlay), (SIZE * 2, y + 25))

    print(
        f"{i+1}/10 | {row['patient_id']} "
        f"| foreground={int(mask.sum())}"
    )

output_path = output_dir / "10_polyp_overlay.png"
canvas.save(output_path)

print("Saved:", output_path)
print("RESULT: PASS")
