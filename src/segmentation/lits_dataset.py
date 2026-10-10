from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LiTSDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        liver_mask_dir,
        tumor_mask_dir,
        image_size=256
    ):
        self.df = dataframe.reset_index(drop=True)

        self.image_dir = Path(image_dir)
        self.liver_mask_dir = Path(liver_mask_dir)
        self.tumor_mask_dir = Path(tumor_mask_dir)

        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Get filenames directly from CSV
        image_name = Path(row["filepath"]).name
        liver_name = Path(row["liver_maskpath"]).name
        tumor_name = Path(row["tumor_maskpath"]).name

        image_path = self.image_dir / image_name
        liver_path = self.liver_mask_dir / liver_name
        tumor_path = self.tumor_mask_dir / tumor_name

        # Read images
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        liver_mask = cv2.imread(
            str(liver_path),
            cv2.IMREAD_GRAYSCALE
        )

        tumor_mask = cv2.imread(
            str(tumor_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if liver_mask is None:
            raise FileNotFoundError(
                f"Liver mask not found: {liver_path}"
            )

        if tumor_mask is None:
            raise FileNotFoundError(
                f"Tumor mask not found: {tumor_path}"
            )

        # Resize
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        liver_mask = cv2.resize(
            liver_mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        tumor_mask = cv2.resize(
            tumor_mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize CT
        image = image.astype(np.float32) / 255.0

        # Binary masks
        liver_mask = (
            liver_mask > 0
        ).astype(np.float32)

        tumor_mask = (
            tumor_mask > 0
        ).astype(np.float32)

        # Convert to tensors
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        liver_mask = torch.tensor(
            liver_mask,
            dtype=torch.float32
        ).unsqueeze(0)

        tumor_mask = torch.tensor(
            tumor_mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return {
            "image": image,
            "liver_mask": liver_mask,
            "tumor_mask": tumor_mask,
            "study_number": int(
                row["study_number"]
            ),
            "instance_number": int(
                row["instance_number"]
            )
        }