import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.segmentation.lits_dataset import LiTSDataset
from src.segmentation.unet import UNet
from src.segmentation.losses import BCEDiceLoss, TumorBCEDiceLoss
from src.segmentation.metrics import (
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "lits_train.csv",
)

TRAIN_SPLIT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "splits",
    "train.csv",
)

VAL_SPLIT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "splits",
    "validation.csv",
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "train",
    "images",
)

LIVER_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "train",
    "liver_masks",
)

TUMOR_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "LiTS",
    "train",
    "tumor_masks",
)

CACHE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "cache",
    "actual_tumor_positivity.csv",
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
)

HISTORY_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "history",
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 256
BATCH_SIZE = 2

EPOCHS = 8

LEARNING_RATE = 5e-5

# Tumor loss settings
TUMOR_POS_WEIGHT = 5.0
TUMOR_LOSS_WEIGHT = 3.0

# Sampling
TRAIN_POSITIVE = 240
TRAIN_NEGATIVE = 60

VAL_POSITIVE = 80
VAL_NEGATIVE = 20

POSITIVE_SAMPLE_WEIGHT = 3.0
NEGATIVE_SAMPLE_WEIGHT = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("IMPROVED TUMOR SEGMENTATION TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Tumor positive pixel weight: {TUMOR_POS_WEIGHT}")
print(f"Tumor loss weight: {TUMOR_LOSS_WEIGHT}")


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_SPLIT)
val_df = pd.read_csv(VAL_SPLIT)

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")


# ============================================================
# LOAD ACTUAL TUMOR POSITIVITY CACHE
# ============================================================

print("\nLoading actual tumor positivity cache...")

cache_df = pd.read_csv(CACHE_PATH)

cache_df["image_name"] = (
    cache_df["image_name"]
    .astype(str)
    .str.strip()
)

cache_lookup = dict(
    zip(
        cache_df["image_name"],
        cache_df["actual_tumor_positive"].astype(bool),
    )
)


def add_actual_tumor_label(df):
    df = df.copy()

    df["mask_filename"] = (
        df["tumor_maskpath"]
        .astype(str)
        .apply(os.path.basename)
    )

    df["actual_tumor_positive"] = (
        df["mask_filename"]
        .map(cache_lookup)
        .fillna(False)
        .astype(bool)
    )

    return df


train_df = add_actual_tumor_label(train_df)
val_df = add_actual_tumor_label(val_df)

print(
    f"Actual train positive slices: "
    f"{train_df['actual_tumor_positive'].sum()}"
)

print(
    f"Actual train negative slices: "
    f"{(~train_df['actual_tumor_positive']).sum()}"
)

print(
    f"Actual validation positive slices: "
    f"{val_df['actual_tumor_positive'].sum()}"
)

print(
    f"Actual validation negative slices: "
    f"{(~val_df['actual_tumor_positive']).sum()}"
)


# ============================================================
# STUDY-LEVEL SPLIT CHECK
# ============================================================

train_studies = set(
    train_df["study_number"].unique()
)

val_studies = set(
    val_df["study_number"].unique()
)

overlap = train_studies.intersection(val_studies)

print("\nStudy-level split:")
print(f"Train studies: {len(train_studies)}")
print(f"Validation studies: {len(val_studies)}")
print(f"Overlap: {len(overlap)}")

if len(overlap) > 0:
    raise RuntimeError(
        "Study leakage detected!"
    )


# ============================================================
# SELECT BALANCED TRAINING DATA
# ============================================================

train_positive = train_df[
    train_df["actual_tumor_positive"]
].copy()

train_negative = train_df[
    ~train_df["actual_tumor_positive"]
].copy()

val_positive = val_df[
    val_df["actual_tumor_positive"]
].copy()

val_negative = val_df[
    ~val_df["actual_tumor_positive"]
].copy()


train_positive = train_positive.sample(
    n=min(TRAIN_POSITIVE, len(train_positive)),
    random_state=SEED,
)

train_negative = train_negative.sample(
    n=min(TRAIN_NEGATIVE, len(train_negative)),
    random_state=SEED,
)

val_positive = val_positive.sample(
    n=min(VAL_POSITIVE, len(val_positive)),
    random_state=SEED,
)

val_negative = val_negative.sample(
    n=min(VAL_NEGATIVE, len(val_negative)),
    random_state=SEED,
)


train_selected = pd.concat(
    [train_positive, train_negative],
    ignore_index=True,
)

val_selected = pd.concat(
    [val_positive, val_negative],
    ignore_index=True,
)

train_selected = train_selected.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)

val_selected = val_selected.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)


print("\nSelected training data:")
print(
    f"Positive: "
    f"{train_selected['actual_tumor_positive'].sum()}"
)

print(
    f"Negative: "
    f"{(~train_selected['actual_tumor_positive']).sum()}"
)

print(f"Total: {len(train_selected)}")

print("\nSelected validation data:")
print(
    f"Positive: "
    f"{val_selected['actual_tumor_positive'].sum()}"
)

print(
    f"Negative: "
    f"{(~val_selected['actual_tumor_positive']).sum()}"
)

print(f"Total: {len(val_selected)}")


# ============================================================
# DATASETS
# ============================================================

print("\nCreating datasets...")

train_dataset = LiTSDataset(
    train_selected,
    IMAGE_DIR,
    LIVER_MASK_DIR,
    TUMOR_MASK_DIR,
    image_size=IMAGE_SIZE,
)

val_dataset = LiTSDataset(
    val_selected,
    IMAGE_DIR,
    LIVER_MASK_DIR,
    TUMOR_MASK_DIR,
    image_size=IMAGE_SIZE,
)


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

sample_weights = np.where(
    train_selected["actual_tumor_positive"].values,
    POSITIVE_SAMPLE_WEIGHT,
    NEGATIVE_SAMPLE_WEIGHT,
)

sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating U-Net...")

model = UNet().to(DEVICE)


# ============================================================
# LOSSES
# ============================================================

liver_loss_fn = BCEDiceLoss()

tumor_loss_fn = TumorBCEDiceLoss(
    pos_weight=TUMOR_POS_WEIGHT,
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
)


# ============================================================
# HELPER
# ============================================================

def calculate_positive_metrics(
    predictions,
    targets,
    positive_flags,
):
    positive_flags = positive_flags.bool()

    if positive_flags.sum().item() == 0:
        return 0.0, 0.0, 0.0, 0.0

    predictions = predictions[positive_flags]
    targets = targets[positive_flags]

    return (
        dice_coefficient(
            predictions,
            targets,
        ).item(),
        iou_score(
            predictions,
            targets,
        ).item(),
        precision_score(
            predictions,
            targets,
        ).item(),
        recall_score(
            predictions,
            targets,
        ).item(),
    )


# ============================================================
# HISTORY
# ============================================================

history = []

best_tumor_positive_dice = -1.0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(1, EPOCHS + 1):

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):

        images = batch["image"].to(DEVICE)
        liver_targets = batch["liver_mask"].to(DEVICE)
        tumor_targets = batch["tumor_mask"].to(DEVICE)

        optimizer.zero_grad()

        liver_logits, tumor_logits = model(images)

        liver_loss = liver_loss_fn(
            liver_logits,
            liver_targets,
        )

        tumor_loss = tumor_loss_fn(
            tumor_logits,
            tumor_targets,
        )

        loss = (
            liver_loss
            + TUMOR_LOSS_WEIGHT * tumor_loss
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = (
        running_loss / len(train_loader)
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    liver_dice_values = []
    liver_iou_values = []
    liver_precision_values = []
    liver_recall_values = []

    tumor_dice_values = []
    tumor_iou_values = []
    tumor_precision_values = []
    tumor_recall_values = []

    tumor_positive_dice_values = []
    tumor_positive_iou_values = []
    tumor_positive_precision_values = []
    tumor_positive_recall_values = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)

            liver_targets = batch[
                "liver_mask"
            ].to(DEVICE)

            tumor_targets = batch[
                "tumor_mask"
            ].to(DEVICE)

            liver_logits, tumor_logits = model(
                images
            )

            liver_loss = liver_loss_fn(
                liver_logits,
                liver_targets,
            )

            tumor_loss = tumor_loss_fn(
                tumor_logits,
                tumor_targets,
            )

            loss = (
                liver_loss
                + TUMOR_LOSS_WEIGHT * tumor_loss
            )

            validation_loss += loss.item()

            # Liver metrics
            liver_dice_values.append(
                dice_coefficient(
                    liver_logits,
                    liver_targets,
                ).item()
            )

            liver_iou_values.append(
                iou_score(
                    liver_logits,
                    liver_targets,
                ).item()
            )

            liver_precision_values.append(
                precision_score(
                    liver_logits,
                    liver_targets,
                ).item()
            )

            liver_recall_values.append(
                recall_score(
                    liver_logits,
                    liver_targets,
                ).item()
            )

            # Tumor metrics
            tumor_dice_values.append(
                dice_coefficient(
                    tumor_logits,
                    tumor_targets,
                ).item()
            )

            tumor_iou_values.append(
                iou_score(
                    tumor_logits,
                    tumor_targets,
                ).item()
            )

            tumor_precision_values.append(
                precision_score(
                    tumor_logits,
                    tumor_targets,
                ).item()
            )

            tumor_recall_values.append(
                recall_score(
                    tumor_logits,
                    tumor_targets,
                ).item()
            )

            # Positive-slice metrics
            positive_flags = (
                tumor_targets
                .view(tumor_targets.size(0), -1)
                .sum(dim=1)
                > 0
            )

            (
                pos_dice,
                pos_iou,
                pos_precision,
                pos_recall,
            ) = calculate_positive_metrics(
                tumor_logits,
                tumor_targets,
                positive_flags,
            )

            if positive_flags.sum().item() > 0:

                tumor_positive_dice_values.append(
                    pos_dice
                )

                tumor_positive_iou_values.append(
                    pos_iou
                )

                tumor_positive_precision_values.append(
                    pos_precision
                )

                tumor_positive_recall_values.append(
                    pos_recall
                )

    validation_loss /= len(val_loader)

    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    liver_dice = np.mean(
        liver_dice_values
    )

    liver_iou = np.mean(
        liver_iou_values
    )

    liver_precision = np.mean(
        liver_precision_values
    )

    liver_recall = np.mean(
        liver_recall_values
    )

    tumor_dice = np.mean(
        tumor_dice_values
    )

    tumor_iou = np.mean(
        tumor_iou_values
    )

    tumor_precision = np.mean(
        tumor_precision_values
    )

    tumor_recall = np.mean(
        tumor_recall_values
    )

    tumor_positive_dice = np.mean(
        tumor_positive_dice_values
    )

    tumor_positive_iou = np.mean(
        tumor_positive_iou_values
    )

    tumor_positive_precision = np.mean(
        tumor_positive_precision_values
    )

    tumor_positive_recall = np.mean(
        tumor_positive_recall_values
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        tumor_positive_dice
    )

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nRESULTS")

    print(
        f"Train loss:       {train_loss:.4f}"
    )

    print(
        f"Validation loss:  {validation_loss:.4f}"
    )

    print("\nLIVER")

    print(
        f"Dice:      {liver_dice:.4f}"
    )

    print(
        f"IoU:       {liver_iou:.4f}"
    )

    print(
        f"Precision: {liver_precision:.4f}"
    )

    print(
        f"Recall:    {liver_recall:.4f}"
    )

    print("\nTUMOR - ALL SLICES")

    print(
        f"Dice:      {tumor_dice:.4f}"
    )

    print(
        f"IoU:       {tumor_iou:.4f}"
    )

    print(
        f"Precision: {tumor_precision:.4f}"
    )

    print(
        f"Recall:    {tumor_recall:.4f}"
    )

    print("\nTUMOR - POSITIVE SLICES ONLY")

    print(
        f"Dice:      {tumor_positive_dice:.4f}"
    )

    print(
        f"IoU:       {tumor_positive_iou:.4f}"
    )

    print(
        f"Precision: {tumor_positive_precision:.4f}"
    )

    print(
        f"Recall:    {tumor_positive_recall:.4f}"
    )

    print(
        f"\nLearning rate: {current_lr:.7f}"
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "liver_dice": liver_dice,
            "liver_iou": liver_iou,
            "liver_precision": liver_precision,
            "liver_recall": liver_recall,
            "tumor_dice": tumor_dice,
            "tumor_iou": tumor_iou,
            "tumor_precision": tumor_precision,
            "tumor_recall": tumor_recall,
            "tumor_positive_dice": tumor_positive_dice,
            "tumor_positive_iou": tumor_positive_iou,
            "tumor_positive_precision": tumor_positive_precision,
            "tumor_positive_recall": tumor_positive_recall,
            "learning_rate": current_lr,
        }
    )

    # --------------------------------------------------------
    # SAVE LAST
    # --------------------------------------------------------

    last_path = os.path.join(
        CHECKPOINT_DIR,
        "last_improved_tumor_unet.pth",
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tumor_positive_dice": tumor_positive_dice,
        },
        last_path,
    )

    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if (
        tumor_positive_dice
        > best_tumor_positive_dice
    ):

        best_tumor_positive_dice = (
            tumor_positive_dice
        )

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best_improved_tumor_unet.pth",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "tumor_positive_dice":
                    tumor_positive_dice,
            },
            best_path,
        )

        print(
            "\nNEW BEST MODEL SAVED"
        )

        print(
            f"Best tumor-positive Dice: "
            f"{best_tumor_positive_dice:.4f}"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = os.path.join(
    HISTORY_DIR,
    "improved_tumor_training_history.csv",
)

history_df = pd.DataFrame(history)

history_df.to_csv(
    history_path,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best tumor-positive Dice: "
    f"{best_tumor_positive_dice:.4f}"
)

print(
    f"\nBest checkpoint:\n"
    f"{os.path.join(CHECKPOINT_DIR, 'best_improved_tumor_unet.pth')}"
)

print(
    f"\nHistory:\n"
    f"{history_path}"
)