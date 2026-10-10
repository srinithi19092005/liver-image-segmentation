import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

VALIDATION_CSV = os.path.join(
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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
    "best_improved_tumor_unet.pth",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "evaluation",
)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "improved_evaluation_results.csv",
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "improved_evaluation_summary.txt",
)

IMAGE_SIZE = 256
BATCH_SIZE = 2


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("IMPROVED MODEL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

print(
    f"Model:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

validation_df = pd.read_csv(
    VALIDATION_CSV
)

print(
    f"Validation rows: "
    f"{len(validation_df)}"
)


# ============================================================
# LOAD ACTUAL TUMOR POSITIVITY CACHE
# ============================================================

cache_df = pd.read_csv(
    CACHE_PATH
)

print(
    f"Cached rows: "
    f"{len(cache_df)}"
)

print(
    "\nValidation columns:"
)

print(
    list(validation_df.columns)
)

print(
    "\nCache columns:"
)

print(
    list(cache_df.columns)
)


# ============================================================
# MATCH VALIDATION MASKS WITH CACHE
# ============================================================

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

validation_df["mask_filename"] = (
    validation_df["tumor_maskpath"]
    .astype(str)
    .apply(os.path.basename)
)

validation_df["actual_tumor_positive"] = (
    validation_df["mask_filename"]
    .map(cache_lookup)
)


# ============================================================
# CHECK MATCHING
# ============================================================

missing_matches = (
    validation_df["actual_tumor_positive"]
    .isna()
    .sum()
)

print(
    "\nMATCHING TUMOR MASKS"
)

if missing_matches > 0:

    print(
        f"WARNING: {missing_matches} "
        f"validation masks were not found in cache."
    )

    raise RuntimeError(
        "Tumor mask cache matching failed."
    )

else:

    print(
        "All validation tumor masks "
        "matched successfully."
    )


validation_df["actual_tumor_positive"] = (
    validation_df["actual_tumor_positive"]
    .astype(bool)
)


positive_count = int(
    validation_df[
        "actual_tumor_positive"
    ].sum()
)

negative_count = (
    len(validation_df) - positive_count
)

print(
    f"Positive validation slices: "
    f"{positive_count}"
)

print(
    f"Negative validation slices: "
    f"{negative_count}"
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION DATASET")
print("=" * 70)

dataset = LiTSDataset(
    validation_df,
    IMAGE_DIR,
    LIVER_MASK_DIR,
    TUMOR_MASK_DIR,
    image_size=IMAGE_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(
    f"Validation batches: "
    f"{len(loader)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = UNet().to(
    DEVICE
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================
#
# IMPORTANT:
# weights_only=False is required because this checkpoint
# contains the complete training checkpoint dictionary.
#
# This checkpoint was created by our own training script.
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)


# ============================================================
# LOAD MODEL STATE
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "tumor_positive_dice" in checkpoint:

        print(
            "Training tumor-positive Dice: "
            f"{checkpoint['tumor_positive_dice']:.4f}"
        )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

liver_loss_fn = BCEDiceLoss()

tumor_loss_fn = TumorBCEDiceLoss(
    pos_weight=5.0
)


# ============================================================
# METRIC STORAGE
# ============================================================

validation_losses = []

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


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING EVALUATION")
print("=" * 70)

with torch.no_grad():

    for batch_index, batch in enumerate(loader):

        images = batch["image"].to(
            DEVICE
        )

        liver_targets = batch[
            "liver_mask"
        ].to(DEVICE)

        tumor_targets = batch[
            "tumor_mask"
        ].to(DEVICE)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        liver_logits, tumor_logits = model(
            images
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        liver_loss = liver_loss_fn(
            liver_logits,
            liver_targets,
        )

        tumor_loss = tumor_loss_fn(
            tumor_logits,
            tumor_targets,
        )

        total_loss = (
            liver_loss
            + 3.0 * tumor_loss
        )

        validation_losses.append(
            total_loss.item()
        )

        # ----------------------------------------------------
        # LIVER METRICS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TUMOR - ALL SLICES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TUMOR - POSITIVE SLICES ONLY
        # ----------------------------------------------------

        positive_flags = (
            tumor_targets
            .view(
                tumor_targets.size(0),
                -1,
            )
            .sum(dim=1)
            > 0
        )

        if positive_flags.sum().item() > 0:

            positive_tumor_logits = (
                tumor_logits[
                    positive_flags
                ]
            )

            positive_tumor_targets = (
                tumor_targets[
                    positive_flags
                ]
            )

            tumor_positive_dice_values.append(
                dice_coefficient(
                    positive_tumor_logits,
                    positive_tumor_targets,
                ).item()
            )

            tumor_positive_iou_values.append(
                iou_score(
                    positive_tumor_logits,
                    positive_tumor_targets,
                ).item()
            )

            tumor_positive_precision_values.append(
                precision_score(
                    positive_tumor_logits,
                    positive_tumor_targets,
                ).item()
            )

            tumor_positive_recall_values.append(
                recall_score(
                    positive_tumor_logits,
                    positive_tumor_targets,
                ).item()
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (batch_index + 1) % 500 == 0
            or batch_index == 0
        ):

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(loader)} batches"
            )


# ============================================================
# FINAL METRICS
# ============================================================

validation_loss = np.mean(
    validation_losses
)

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


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nValidation loss: "
    f"{validation_loss:.4f}"
)

print("\nLIVER")

print(
    f"  Dice:      {liver_dice:.4f}"
)

print(
    f"  IoU:       {liver_iou:.4f}"
)

print(
    f"  Precision: {liver_precision:.4f}"
)

print(
    f"  Recall:    {liver_recall:.4f}"
)

print("\nTUMOR - ALL SLICES")

print(
    f"  Dice:      {tumor_dice:.4f}"
)

print(
    f"  IoU:       {tumor_iou:.4f}"
)

print(
    f"  Precision: {tumor_precision:.4f}"
)

print(
    f"  Recall:    {tumor_recall:.4f}"
)

print("\nTUMOR - POSITIVE SLICES ONLY")

print(
    f"  Dice:      {tumor_positive_dice:.4f}"
)

print(
    f"  IoU:       {tumor_positive_iou:.4f}"
)

print(
    f"  Precision: {tumor_positive_precision:.4f}"
)

print(
    f"  Recall:    {tumor_positive_recall:.4f}"
)

print(
    f"  Positive slices: "
    f"{positive_count}"
)


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results = {
    "model": [
        "best_improved_tumor_unet.pth"
    ],
    "validation_loss": [
        validation_loss
    ],
    "liver_dice": [
        liver_dice
    ],
    "liver_iou": [
        liver_iou
    ],
    "liver_precision": [
        liver_precision
    ],
    "liver_recall": [
        liver_recall
    ],
    "tumor_dice_all": [
        tumor_dice
    ],
    "tumor_iou_all": [
        tumor_iou
    ],
    "tumor_precision_all": [
        tumor_precision
    ],
    "tumor_recall_all": [
        tumor_recall
    ],
    "tumor_positive_dice": [
        tumor_positive_dice
    ],
    "tumor_positive_iou": [
        tumor_positive_iou
    ],
    "tumor_positive_precision": [
        tumor_positive_precision
    ],
    "tumor_positive_recall": [
        tumor_positive_recall
    ],
    "positive_validation_slices": [
        positive_count
    ],
}

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_CSV,
    index=False,
)

print(
    "\nResults saved:"
)

print(
    RESULTS_CSV
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = f"""
IMPROVED TUMOR U-NET EVALUATION
================================

Model:
{MODEL_PATH}

Device:
{DEVICE}

Validation slices:
{len(validation_df)}

Positive validation slices:
{positive_count}

Negative validation slices:
{negative_count}

Validation loss:
{validation_loss:.4f}


LIVER
-----

Dice:
{liver_dice:.4f}

IoU:
{liver_iou:.4f}

Precision:
{liver_precision:.4f}

Recall:
{liver_recall:.4f}


TUMOR - ALL SLICES
------------------

Dice:
{tumor_dice:.4f}

IoU:
{tumor_iou:.4f}

Precision:
{tumor_precision:.4f}

Recall:
{tumor_recall:.4f}


TUMOR - POSITIVE SLICES ONLY
----------------------------

Dice:
{tumor_positive_dice:.4f}

IoU:
{tumor_positive_iou:.4f}

Precision:
{tumor_positive_precision:.4f}

Recall:
{tumor_positive_recall:.4f}
"""

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(summary)


print(
    "\nSummary saved:"
)

print(
    SUMMARY_PATH
)


# ============================================================
# SAVE VISUAL EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SAVING VISUAL EXAMPLES")
print("=" * 70)

model.eval()

example_count = 0

with torch.no_grad():

    for batch in loader:

        images = batch["image"].to(
            DEVICE
        )

        liver_logits, tumor_logits = model(
            images
        )

        liver_predictions = (
            torch.sigmoid(
                liver_logits
            ) > 0.5
        ).float()

        tumor_predictions = (
            torch.sigmoid(
                tumor_logits
            ) > 0.5
        ).float()

        for i in range(
            images.size(0)
        ):

            if example_count >= 5:
                break

            image = (
                images[i, 0]
                .cpu()
                .numpy()
            )

            liver_prediction = (
                liver_predictions[
                    i, 0
                ]
                .cpu()
                .numpy()
            )

            tumor_prediction = (
                tumor_predictions[
                    i, 0
                ]
                .cpu()
                .numpy()
            )

            liver_target = (
                batch["liver_mask"][
                    i, 0
                ]
                .numpy()
            )

            tumor_target = (
                batch["tumor_mask"][
                    i, 0
                ]
                .numpy()
            )

            image_uint8 = (
                image * 255
            ).astype(np.uint8)

            liver_pred_uint8 = (
                liver_prediction * 255
            ).astype(np.uint8)

            tumor_pred_uint8 = (
                tumor_prediction * 255
            ).astype(np.uint8)

            liver_target_uint8 = (
                liver_target * 255
            ).astype(np.uint8)

            tumor_target_uint8 = (
                tumor_target * 255
            ).astype(np.uint8)

            # ------------------------------------------------
            # OVERLAY
            # ------------------------------------------------

            overlay = cv2.cvtColor(
                image_uint8,
                cv2.COLOR_GRAY2BGR,
            )

            liver_pixels = (
                liver_prediction > 0.5
            )

            tumor_pixels = (
                tumor_prediction > 0.5
            )

            overlay[
                liver_pixels
            ] = [0, 255, 0]

            overlay[
                tumor_pixels
            ] = [0, 0, 255]

            # ------------------------------------------------
            # CONVERT MASKS TO COLOR
            # ------------------------------------------------

            liver_target_bgr = cv2.cvtColor(
                liver_target_uint8,
                cv2.COLOR_GRAY2BGR,
            )

            tumor_target_bgr = cv2.cvtColor(
                tumor_target_uint8,
                cv2.COLOR_GRAY2BGR,
            )

            liver_pred_bgr = cv2.cvtColor(
                liver_pred_uint8,
                cv2.COLOR_GRAY2BGR,
            )

            tumor_pred_bgr = cv2.cvtColor(
                tumor_pred_uint8,
                cv2.COLOR_GRAY2BGR,
            )

            # ------------------------------------------------
            # LABEL FUNCTION
            # ------------------------------------------------

            def add_label(
                img,
                text,
            ):

                result = img.copy()

                cv2.rectangle(
                    result,
                    (0, 0),
                    (
                        result.shape[1],
                        30,
                    ),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    result,
                    text,
                    (5, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                return result

            # ------------------------------------------------
            # SIX PANELS
            # ------------------------------------------------

            panels = [

                add_label(
                    cv2.cvtColor(
                        image_uint8,
                        cv2.COLOR_GRAY2BGR,
                    ),
                    "Input",
                ),

                add_label(
                    liver_target_bgr,
                    "Liver Ground Truth",
                ),

                add_label(
                    liver_pred_bgr,
                    "Liver Prediction",
                ),

                add_label(
                    tumor_target_bgr,
                    "Tumor Ground Truth",
                ),

                add_label(
                    tumor_pred_bgr,
                    "Tumor Prediction",
                ),

                add_label(
                    overlay,
                    "Prediction Overlay",
                ),
            ]

            # ------------------------------------------------
            # CREATE GRID
            # ------------------------------------------------

            top_row = np.hstack(
                panels[:3]
            )

            bottom_row = np.hstack(
                panels[3:]
            )

            final_image = np.vstack(
                [
                    top_row,
                    bottom_row,
                ]
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"improved_prediction_example_{example_count + 1}.png",
            )

            cv2.imwrite(
                output_path,
                final_image,
            )

            print(
                output_path
            )

            example_count += 1

        if example_count >= 5:
            break


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED EVALUATION COMPLETE")
print("=" * 70)