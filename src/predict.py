import io
import time
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.model import build_model
from src.metrics import to_prediction

IMAGE_SIZE = (256, 448)


def load_model(weights_path, device):
    """Load the trained model once, ready for repeated predictions"""
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_bytes(model, image_bytes, device, return_mask=False):
    """Run the model on an uploaded image and return summary numbers"""
    start = time.time()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size

    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    x = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)["out"]
        probs = torch.softmax(out, dim=1)[0, 1]  # fish probability per pixel
        mask = to_prediction(out).cpu()[0]

    fish_pixels = int(mask.sum())
    total_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]

    if fish_pixels > 0:
        confidence = float(probs[mask == 1].mean())
    else:
        confidence = None

    result = {
        "fish_detected": fish_pixels > 0,
        "fish_pixels": fish_pixels,
        "fish_coverage_percent": round((fish_pixels / total_pixels) * 100, 3),
        "mean_confidence": round(confidence, 3) if confidence is not None else None,
        "original_size": list(original_size),
        "inference_ms": round((time.time() - start) * 1000, 1),
    }

    # Optinal: return the image and mask for visualization
    if return_mask:
        result["_image"] = image
        result["_mask"] = mask.numpy()

    return result
