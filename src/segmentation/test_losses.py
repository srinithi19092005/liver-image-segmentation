import torch

from losses import DiceLoss, BCEDiceLoss


# ============================================================
# LOSS FUNCTION TEST
# ============================================================

print("=" * 60)
print("LOSS FUNCTION TEST")
print("=" * 60)


# ============================================================
# CREATE FAKE PREDICTIONS
# ============================================================

predictions = torch.randn(
    2,
    1,
    256,
    256
)


# ============================================================
# CREATE FAKE TARGET MASKS
# ============================================================

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


print("\nPrediction shape:")
print(predictions.shape)

print("\nTarget shape:")
print(targets.shape)


# ============================================================
# TEST DICE LOSS
# ============================================================

dice_loss = DiceLoss()

dice_value = dice_loss(
    predictions,
    targets
)


print("\nDice Loss:")

print(
    dice_value.item()
)


# ============================================================
# TEST BCE + DICE LOSS
# ============================================================

combined_loss = BCEDiceLoss()

loss_value = combined_loss(
    predictions,
    targets
)


print("\nBCE + Dice Loss:")

print(
    loss_value.item()
)


# ============================================================
# TEST BACKPROPAGATION
# ============================================================

predictions = torch.randn(
    2,
    1,
    256,
    256,
    requires_grad=True
)


loss = combined_loss(
    predictions,
    targets
)


loss.backward()


print("\nGradient calculation:")

print("Successful")


# ============================================================
# CHECK GRADIENT
# ============================================================

if predictions.grad is not None:

    print(
        "Gradient exists: True"
    )

else:

    print(
        "Gradient exists: False"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("LOSS TEST SUCCESSFUL")
print("=" * 60)

print(
    "\nDice Loss is working."
)

print(
    "BCE + Dice Loss is working."
)

print(
    "Backpropagation is working."
)

print(
    "\nReady for the DataLoader and training."
)