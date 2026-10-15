
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from preprocess import load_sample


# ==========================================
# 1. ALLOWED DATA SPLITS
# ==========================================

ALLOWED_SPLITS = {
    "SEG_TRAIN",
    "SEG_VAL",
    "META_FIT",
    "CALIBRATION",
    "OUTER_TEST",
    "NEVER_POLYP_TEST",
}


# ==========================================
# 2. IMAGENET NORMALIZATION
# ==========================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ==========================================
# 3. FAR-POLYP-SEG DATASET
# ==========================================

class FARPolypDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        patient_ids,
        split,
        augment=False,
        image_ids=None
    ):

        # Validate split
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Invalid split: {split}"
            )

        # Augmentation only for training
        if augment and split != "SEG_TRAIN":
            raise ValueError(
                "Augmentation is only allowed for SEG_TRAIN"
            )

        # Normalize patient IDs
        patient_ids = {
            str(p) for p in patient_ids
        }

        if not patient_ids:
            raise ValueError(
                "patient_ids cannot be empty"
            )

        # Read dataset manifest
        df = pd.read_csv(manifest_path)

        # Filter by patient IDs
        self.df = df[
            df["patient_id"].astype(str).isin(patient_ids)
        ].reset_index(drop=True)

        # ==================================
        # OPTIONAL OFFICIAL IMAGE FILTERING
        # ==================================

        if image_ids is not None:

            image_ids = {
                str(x) for x in image_ids
            }

            if not image_ids:
                raise ValueError(
                    "image_ids cannot be empty"
                )

            available_ids = set(
                self.df["image_id"].astype(str)
            )

            missing_ids = image_ids - available_ids

            if missing_ids:
                raise ValueError(
                    "Unknown image IDs: "
                    f"{sorted(missing_ids)[:5]}"
                )

            self.df = self.df[
                self.df["image_id"].astype(str).isin(image_ids)
            ].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(
                "No images found for selected patients/images"
            )

        self.split = split
        self.augment = augment

    # ======================================
    # 4. DATASET LENGTH
    # ======================================

    def __len__(self):
        return len(self.df)

    # ======================================
    # 5. SYNCHRONIZED AUGMENTATION
    # ======================================

    def apply_augmentation(self, image, mask):

        # ----------------------------------
        # Horizontal Flip - p = 0.5
        # ----------------------------------

        if torch.rand(()).item() < 0.5:

            image = np.flip(
                image, axis=1
            ).copy()

            mask = np.flip(
                mask, axis=1
            ).copy()

        # ----------------------------------
        # Vertical Flip - p = 0.5
        # ----------------------------------

        if torch.rand(()).item() < 0.5:

            image = np.flip(
                image, axis=0
            ).copy()

            mask = np.flip(
                mask, axis=0
            ).copy()

        # ----------------------------------
        # Random Rotation
        # Angle: [-15, +15]
        # Probability: 0.3
        # ----------------------------------

        if torch.rand(()).item() < 0.3:

            angle = (
                torch.rand(()).item() * 30.0
            ) - 15.0

            # Image: float [0,1] -> uint8
            image_uint8 = np.clip(
                image * 255.0,
                0,
                255
            ).astype(np.uint8)

            # Mask: binary 0/1 -> uint8
            mask_uint8 = (
                mask * 255
            ).astype(np.uint8)

            image_pil = Image.fromarray(
                image_uint8,
                mode="RGB"
            )

            mask_pil = Image.fromarray(
                mask_uint8
            )

            # Rotate RGB image: Bilinear
            image_pil = image_pil.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(0, 0, 0)
            )

            # Rotate mask: Nearest Neighbor
            mask_pil = mask_pil.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
                fillcolor=0
            )

            # Convert back to NumPy
            image = np.asarray(
                image_pil,
                dtype=np.float32
            ) / 255.0

            mask = (
                np.asarray(mask_pil) >= 128
            ).astype(np.float32)

        return image, mask

    # ======================================
    # 6. GET DATA SAMPLE
    # ======================================

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ----------------------------------
        # Step 1: Load RGB image and mask
        # ----------------------------------

        # preprocess.py handles:
        # - Resize to 256x256
        # - RGB conversion
        # - Binary mask
        image, mask = load_sample(row)

        # ----------------------------------
        # Step 2: Data augmentation
        # ----------------------------------

        if self.augment:
            image, mask = self.apply_augmentation(
                image, mask
            )

        # ----------------------------------
        # Step 3: ImageNet normalization
        # ----------------------------------

        # Normalize image only, not mask
        image = (
            image - IMAGENET_MEAN
        ) / IMAGENET_STD

        image = image.astype(
            np.float32
        )

        mask = mask.astype(
            np.float32
        )

        # ----------------------------------
        # Step 4: HWC -> CHW
        # ----------------------------------

        image = np.transpose(
            image, (2, 0, 1)
        ).copy()

        # Mask: HW -> 1HW
        mask = np.expand_dims(
            mask, axis=0
        ).copy()

        # ----------------------------------
        # Step 5: PyTorch Tensors
        # ----------------------------------

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # ----------------------------------
        # Step 6: Return sample
        # ----------------------------------

        return {
            "image": image,
            "mask": mask,
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
            "split": self.split,
        }
