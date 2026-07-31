import torch
from src.metrics import to_prediction, iou_score


def test_identical_masks_score_one():
    mask = torch.zeros(1, 4, 4)
    mask[0, 1:3, 1:3] = 1
    assert iou_score(mask, mask) == 1.0


def test_no_overlap_score_zero():
    m1 = torch.zeros(1, 4, 4)
    m1[0, 1:3, 1:3] = 1

    m2 = torch.zeros(1, 4, 4)
    m2[0, 0, 0] = 1

    assert iou_score(m1, m2) == 0.0


def test_half_overlap_scores_half():
    m1 = torch.zeros(1, 4, 4)
    m1[0, 1:3, 1:3] = 1

    m2 = torch.zeros(1, 4, 4)
    m2[0, 1:3, 1:2] = 1

    assert iou_score(m1, m2) == 0.5
