import torch


def _prepare_predictions(predictions):
    """
    Convert model logits/probabilities into binary predictions.

    If values are outside [0, 1], they are treated as logits
    and sigmoid is applied.
    """

    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    return (predictions > 0.5).float()


def dice_coefficient(predictions, targets, smooth=1.0):
    """
    Calculate Dice coefficient.

    Handles empty masks:
    - empty target + empty prediction -> 1
    - empty target + non-empty prediction -> 0
    """

    predictions = _prepare_predictions(predictions)

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.float().view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum +
        target_sum +
        smooth
    )

    return dice.mean()


def iou_score(predictions, targets, smooth=1.0):
    """
    Calculate Intersection over Union.
    """

    predictions = _prepare_predictions(predictions)

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.float().view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions +
        targets -
        predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()


def precision_score(predictions, targets, smooth=1.0):
    """
    Calculate precision.
    """

    predictions = _prepare_predictions(predictions)

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.float().view(
        targets.size(0), -1
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    predicted_positive = predictions.sum(
        dim=1
    )

    precision = (
        true_positive + smooth
    ) / (
        predicted_positive + smooth
    )

    return precision.mean()


def recall_score(predictions, targets, smooth=1.0):
    """
    Calculate recall.
    """

    predictions = _prepare_predictions(predictions)

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.float().view(
        targets.size(0), -1
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    actual_positive = targets.sum(
        dim=1
    )

    recall = (
        true_positive + smooth
    ) / (
        actual_positive + smooth
    )

    return recall.mean()