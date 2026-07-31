import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from src.model import build_model
from src.metrics import to_prediction, iou_score
from tests.conftest import needs_data, needs_model

SEG = Path("data/Segmentation")
WEIGHTS = Path("models/best_model.pth")

MIN_MEAN_IOU = 0.70             # measured 0.786 on the full test set
MAX_EMPTY_FALSE_PIXELS = 100    # measured 0 across all 90 empty images


@pytest.fixture(scope="module")
def model():
    m = build_model(num_classes=2)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()
    return m


def load_ids():
    ids = []
    for line in open(Path(__file__).parent / "regression_images.txt"):
        ids.append(line.strip())
    return ids


def predict(model, image_id):
    image = Image.open(SEG / "images" / f"{image_id}.jpg").convert("RGB")
    image = image.resize((448, 256))
    x = TF.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        return to_prediction(model(x)["out"])[0]


@needs_data
@needs_model
def test_mean_iou_above_threshold(model):
    scores = []
    for image_id in load_ids():
        truth = Image.open(SEG / "masks" / f"{image_id}.png").convert("L")
        truth = truth.resize((448, 256), Image.NEAREST)
        truth = torch.from_numpy((np.array(truth) > 127).astype("int64"))

        if truth.sum() == 0:
            continue  # IOU is meaningless for empty images

        pred = predict(model, image_id)
        scores.append(iou_score(pred.unsqueeze(0), truth.unsqueeze(0)))

    mean_iou = float(np.mean(scores))
    assert mean_iou >= MIN_MEAN_IOU, f"Mean IoU dropped to {mean_iou:.3f}"


@needs_data
@needs_model
def test_no_false_positives_on_empty_images(model):
    for image_id in load_ids():
        truth = Image.open(SEG / "masks" / f"{image_id}.png").convert("L")
        truth = truth.resize((448, 256), Image.NEAREST)

        if (np.array(truth) > 127).sum() > 0:
            continue  # only test empty images

        pred = predict(model, image_id)
        assert (
            int(pred.sum()) <= MAX_EMPTY_FALSE_PIXELS
        ), f"{image_id} has {int(pred.sum())} false positive pixels"
