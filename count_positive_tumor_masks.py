import cv2
from pathlib import Path

mask_dir = Path(r"dataset\LiTS\train\tumor_masks")

files = list(mask_dir.glob("*.png"))

total = len(files)
positive = 0

print("Checking tumor masks...")
print("Total files:", total)

for i, path in enumerate(files):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        continue

    if (mask > 0).any():
        positive += 1

    if (i + 1) % 5000 == 0:
        print(
            "Checked:",
            i + 1,
            "| positive:",
            positive
        )

print()
print("=" * 50)
print("FINAL RESULT")
print("=" * 50)
print("Total masks:", total)
print("Masks with positive pixels:", positive)
print("Empty masks:", total - positive)
print("=" * 50)