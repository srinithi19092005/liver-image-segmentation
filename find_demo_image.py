import os
import glob
import cv2

PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "train",
    "images",
)

LIVER_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "train",
    "liver_masks",
)

TUMOR_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "train",
    "tumor_masks",
)

print("=" * 70)
print("SEARCHING FOR GOOD DEMONSTRATION IMAGES")
print("=" * 70)

tumor_files = glob.glob(
    os.path.join(TUMOR_DIR, "*.png")
)

found = 0

for tumor_path in tumor_files:

    tumor_filename = os.path.basename(
        tumor_path
    )

    liver_filename = tumor_filename.replace(
        "lesionmask",
        "livermask",
    )

    liver_path = os.path.join(
        LIVER_DIR,
        liver_filename,
    )

    if not os.path.exists(liver_path):
        continue

    tumor_mask = cv2.imread(
        tumor_path,
        cv2.IMREAD_GRAYSCALE,
    )

    liver_mask = cv2.imread(
        liver_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if tumor_mask is None or liver_mask is None:
        continue

    tumor_pixels = int(
        (tumor_mask > 0).sum()
    )

    liver_pixels = int(
        (liver_mask > 0).sum()
    )

    if tumor_pixels > 300 and liver_pixels > 5000:

        image_filename = tumor_filename.replace(
            "segmentation-101_lesionmask_",
            "volume-101_",
        )

        image_path = os.path.join(
            IMAGE_DIR,
            image_filename,
        )

        if not os.path.exists(image_path):
            continue

        found += 1

        print()
        print("DEMO IMAGE #", found)
        print("CT image:", image_filename)
        print("Full path:", image_path)
        print("Tumor pixels:", tumor_pixels)
        print("Liver pixels:", liver_pixels)
        print("-" * 70)

        if found >= 10:
            break

print()

if found == 0:
    print("No suitable demonstration images were found.")
else:
    print(
        "Found",
        found,
        "suitable demonstration images."
    )

print("=" * 70)