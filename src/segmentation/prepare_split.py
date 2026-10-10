from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80

VALIDATION_RATIO = 0.20


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path.cwd()

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "LiTS"
)

CSV_PATH = (
    DATASET_DIR
    / "lits_train.csv"
)

OUTPUT_DIR = (
    DATASET_DIR
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_CSV = (
    OUTPUT_DIR
    / "train.csv"
)

VALIDATION_CSV = (
    OUTPUT_DIR
    / "validation.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("LiTS STUDY-LEVEL DATASET SPLIT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading CSV:")

print(CSV_PATH)


if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"\nCSV not found:\n{CSV_PATH}"
    )


df = pd.read_csv(
    CSV_PATH
)


print(
    f"\nTotal slices: {len(df):,}"
)


# ============================================================
# CHECK STUDY COLUMN
# ============================================================

if "study_number" not in df.columns:

    raise ValueError(
        "study_number column was not found."
    )


# ============================================================
# GET UNIQUE STUDIES
# ============================================================

studies = (
    df["study_number"]
    .drop_duplicates()
    .tolist()
)


print(
    f"Unique studies: {len(studies):,}"
)


# ============================================================
# SHUFFLE STUDIES
# ============================================================

study_series = pd.Series(
    studies
)

study_series = study_series.sample(
    frac=1.0,
    random_state=SEED
).reset_index(
    drop=True
)


# ============================================================
# SPLIT STUDIES
# ============================================================

train_study_count = int(
    len(study_series)
    * TRAIN_RATIO
)


train_studies = set(
    study_series.iloc[
        :train_study_count
    ]
)


validation_studies = set(
    study_series.iloc[
        train_study_count:
    ]
)


# ============================================================
# CREATE DATAFRAME SPLITS
# ============================================================

train_df = df[
    df["study_number"].isin(
        train_studies
    )
].copy()


validation_df = df[
    df["study_number"].isin(
        validation_studies
    )
].copy()


# ============================================================
# VERIFY NO STUDY OVERLAP
# ============================================================

overlap = (
    set(train_df["study_number"])
    &
    set(validation_df["study_number"])
)


if len(overlap) > 0:

    raise RuntimeError(
        f"Study overlap detected: {overlap}"
    )


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_CSV,
    index=False
)


validation_df.to_csv(
    VALIDATION_CSV,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(
    f"\nTraining studies: "
    f"{len(train_studies):,}"
)

print(
    f"Validation studies: "
    f"{len(validation_studies):,}"
)

print(
    f"\nTraining slices: "
    f"{len(train_df):,}"
)

print(
    f"Validation slices: "
    f"{len(validation_df):,}"
)

print(
    f"\nStudy overlap: "
    f"{len(overlap)}"
)

print("\nTraining CSV:")

print(
    TRAIN_CSV
)

print("\nValidation CSV:")

print(
    VALIDATION_CSV
)

print("\n" + "=" * 70)
print("STUDY-LEVEL SPLIT SUCCESSFUL")
print("=" * 70)