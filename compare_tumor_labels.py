import pandas as pd
import cv2

from pathlib import Path


CSV_PATH = r"dataset\LiTS\lits_train.csv"

TUMOR_MASK_DIR = Path(
    r"dataset\LiTS\train\tumor_masks"
)


df = pd.read_csv(CSV_PATH)

csv_positive = 0
actual_positive = 0
agreement = 0
disagreement = 0

checked = 0

print("=" * 60)
print("CSV vs ACTUAL TUMOR MASK CHECK")
print("=" * 60)

for i in range(len(df)):

    row = df.iloc[i]

    tumor_name = Path(
        row["tumor_maskpath"]
    ).name

    mask_path = TUMOR_MASK_DIR / tumor_name

    if not mask_path.exists():
        continue

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        continue

    csv_is_positive = not bool(
        row["tumor_mask_empty"]
    )

    actual_is_positive = bool(
        (mask > 0).any()
    )

    if csv_is_positive:
        csv_positive += 1

    if actual_is_positive:
        actual_positive += 1

    if csv_is_positive == actual_is_positive:
        agreement += 1
    else:
        disagreement += 1

    checked += 1

print()
print("Rows checked:", checked)
print("CSV positive:", csv_positive)
print("Actual positive:", actual_positive)
print("Agreement:", agreement)
print("Disagreement:", disagreement)

print()
print("=" * 60)