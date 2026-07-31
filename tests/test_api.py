from pathlib import Path
from fastapi.testclient import TestClient
from src.api import app
from tests.conftest import needs_data, needs_model

SEG = Path("data/Segmentation")
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_non_image():
    files = {"file": ("notes.txt", b"This is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


@needs_data
@needs_model
def test_predict_returns_expected_fields():
    with open(SEG / "images" / "valid" / "9862_Acanthopagrus_palmaris_f000090.jpg", "rb") as f:
        files = {"file": ("fish.jpg", f.read(), "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    body = response.json()
    for field in [
        "fish_detected",
        "fish_pixels",
        "fish_coverage_percent",
        "mean_confidence",
        "original_size",
        "inference_ms",
    ]:
        assert field in body
