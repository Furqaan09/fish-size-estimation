import torch


def to_prediction(output):
    """Model output [B, 2, H, W] -> Prediction mask [B, H, W] of 0s and 1s."""
    return output.argmax(dim=1)


def iou_score(pred, target, eps=1e-7):
    """IoU for the fish class between two [B, H, W] masks of 0s and 1s."""
    pred_fish = pred == 1
    true_fish = target == 1

    intersection = (pred_fish & true_fish).sum().float()
    union = (pred_fish | true_fish).sum().float()

    return (intersection / (union + eps)).item()
