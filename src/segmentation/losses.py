import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(
            predictions.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return (
            1.0 - dice
        ).mean()


class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        predictions,
        targets
    ):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


class TumorBCEDiceLoss(nn.Module):

    def __init__(
        self,
        pos_weight=20.0,
        bce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.pos_weight = pos_weight

        self.dice = DiceLoss()

    def forward(
        self,
        predictions,
        targets
    ):

        weight = torch.tensor(
            self.pos_weight,
            device=predictions.device
        )

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=weight
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss