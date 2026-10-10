import pandas as pd
import cv2
from pathlib import Path


CSV_PATH = r"dataset\LiTS\splits\validation.csv"

TUMOR_MASK_DIR = Path(
    r"dataset\LiTS\train\tumor_masks"
)


df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("RAW TUMOR MASK DIAGNOSTIC")
print("=" * 70)

print()
print("Total validation rows:", len(df))

print()
print("First 10 CSV rows:")
print()

for i in range(10):

    row = df.iloc[i]

    tumor_name = Path(
        row["tumor_maskpath"]
    ).name

    tumor_path = TUMOR_MASK_DIR / tumor_name

    print(
        f"{i:02d} | "
        f"CSV tumor_mask_empty={row['tumor_mask_empty']} | "
        f"file={tumor_name}"
    )

    print(
        f"    exists={tumor_path.exists()} | "
        f"path={tumor_path}"
    )

    if tumor_path.exists():

        mask = cv2.imread(
            str(tumor_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            print("    cv2 read: FAILED")

        else:

            print(
                f"    shape={mask.shape} | "
                f"min={mask.min()} | "
                f"max={mask.max()} | "
                f"nonzero_pixels={(mask > 0).sum()} | "
                f"unique_values={len(set(mask.flatten()))}"
            )

    print()


print("=" * 70)
print("SUMMARY OF FIRST 100 MASK FILES")
print("=" * 70)

csv_positive = 0
raw_positive = 0
missing = 0

for i in range(min(100, len(df))):

    row = df.iloc[i]

    if not bool(row["tumor_mask_empty"]):
        csv_positive += 1

    tumor_name = Path(
        row["tumor_maskpath"]
    ).name

    tumor_path = TUMOR_MASK_DIR / tumor_name

    if not tumor_path.exists():

        missing += 1
        continue

    mask = cv2.imread(
        str(tumor_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is not None and (mask > 0).sum() > 0:
        raw_positive += 1


print("CSV positive:", csv_positive)
print("Raw mask positive:", raw_positive)
print("Missing mask files:", missing)

print()
print("=" * 70)

if csv_positive > 0 and raw_positive == 0:

    print(
        "IMPORTANT: CSV says tumor masks are positive, "
        "but the actual mask files contain no positive pixels."
    )

elif raw_positive > 0:

    print(
        "The actual tumor mask files contain positive pixels."
    )

else:

    print(
        "No positive tumor masks were found in the checked files."
    )

print("=" * 70)