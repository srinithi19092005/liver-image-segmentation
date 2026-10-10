import torch

from metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score
)


print("=" * 60)
print("SEGMENTATION METRICS TEST")
print("=" * 60)


# Create fake predictions
predictions = torch.randn(
    2,
    1,
    256,
    256
)


# Create fake binary target masks
targets = torch.randint(
    0,
    2,
    (
        2,
        1,
        256,
        256
    )
).float()


# Calculate metrics
dice = dice_score(
    predictions,
    targets
)

iou = iou_score(
    predictions,
    targets
)

precision = precision_score(
    predictions,
    targets
)

recall = recall_score(
    predictions,
    targets
)


# Display results
print("\nDice:")
print(dice)

print("\nIoU:")
print(iou)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)


print("\n" + "=" * 60)
print("METRICS TEST SUCCESSFUL")
print("=" * 60)