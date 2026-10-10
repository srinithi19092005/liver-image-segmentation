from pathlib import Path
import sys

import pandas as pd
import torch


# ============================================================
# MAKE SURE PYTHON CAN FIND lits_dataset.py
# ============================================================

CURRENT_FOLDER = Path(__file__).parent

sys.path.insert(
    0,
    str(CURRENT_FOLDER)
)


from lits_dataset import LiTSDataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path.cwd()

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "LiTS"
    / "train"
)

CSV_PATH = (
    PROJECT_DIR
    / "dataset"
    / "LiTS"
    / "lits_train.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("LiTS DATASET TEST")
print("=" * 60)


# ============================================================
# CHECK CSV
# ============================================================

print("\nCSV:")
print(CSV_PATH)


if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_PATH}"
    )


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

print("\nChecking dataset directories...")


image_dir = DATASET_DIR / "images"

liver_mask_dir = DATASET_DIR / "liver_masks"

tumor_mask_dir = DATASET_DIR / "tumor_masks"


print("\nImages:")
print(image_dir)

print("\nLiver masks:")
print(liver_mask_dir)

print("\nTumor masks:")
print(tumor_mask_dir)


if not image_dir.exists():

    raise FileNotFoundError(
        f"\nImage directory not found:\n{image_dir}"
    )


if not liver_mask_dir.exists():

    raise FileNotFoundError(
        f"\nLiver mask directory not found:\n{liver_mask_dir}"
    )


if not tumor_mask_dir.exists():

    raise FileNotFoundError(
        f"\nTumor mask directory not found:\n{tumor_mask_dir}"
    )


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading CSV...")

df = pd.read_csv(
    CSV_PATH
)


print(
    f"\nNumber of samples: {len(df):,}"
)


# ============================================================
# SHOW CSV COLUMNS
# ============================================================

print("\nCSV columns:")

for column in df.columns:

    print(
        " -",
        column
    )


# ============================================================
# CREATE DATASET
# ============================================================

print("\nCreating PyTorch Dataset...")


dataset = LiTSDataset(

    dataframe=df,

    image_dir=image_dir,

    liver_mask_dir=liver_mask_dir,

    tumor_mask_dir=tumor_mask_dir,

    image_size=256
)


print(
    f"\nDataset length: {len(dataset):,}"
)


# ============================================================
# LOAD FIRST SAMPLE
# ============================================================

print("\nLoading first sample...")


sample = dataset[0]


# ============================================================
# PRINT SAMPLE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE INFORMATION")
print("=" * 60)


print("\nImage shape:")

print(
    sample["image"].shape
)


print("\nLiver mask shape:")

print(
    sample["liver_mask"].shape
)


print("\nTumor mask shape:")

print(
    sample["tumor_mask"].shape
)


print("\nStudy number:")

print(
    sample["study_number"]
)


print("\nInstance number:")

print(
    sample["instance_number"]
)


# ============================================================
# CHECK IMAGE VALUES
# ============================================================

print("\n" + "=" * 60)
print("IMAGE VALUES")
print("=" * 60)


image_min = (
    sample["image"]
    .min()
    .item()
)

image_max = (
    sample["image"]
    .max()
    .item()
)


print(
    "\nImage range:"
)

print(
    image_min,
    "to",
    image_max
)


# ============================================================
# CHECK LIVER MASK VALUES
# ============================================================

print("\n" + "=" * 60)
print("LIVER MASK")
print("=" * 60)


liver_values = torch.unique(
    sample["liver_mask"]
)


print(
    "\nLiver mask values:"
)

print(
    liver_values
)


# ============================================================
# CHECK TUMOR MASK VALUES
# ============================================================

print("\n" + "=" * 60)
print("TUMOR MASK")
print("=" * 60)


tumor_values = torch.unique(
    sample["tumor_mask"]
)


print(
    "\nTumor mask values:"
)

print(
    tumor_values
)


# ============================================================
# COUNT FOREGROUND PIXELS
# ============================================================

liver_pixels = (
    sample["liver_mask"] > 0
).sum().item()


tumor_pixels = (
    sample["tumor_mask"] > 0
).sum().item()


print(
    "\nLiver foreground pixels:"
)

print(
    liver_pixels
)


print(
    "\nTumor foreground pixels:"
)

print(
    tumor_pixels
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATASET TEST SUCCESSFUL")
print("=" * 60)

print(
    "\nYour PyTorch Dataset is working correctly."
)

print(
    "\nReady for the DataLoader and U-Net."
)