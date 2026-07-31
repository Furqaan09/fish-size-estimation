import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from src.predict import load_model, predict_bytes

app = FastAPI(title="Fish Segmentation API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"

_model = None


def get_model():
    """Load the model once, on first use rather than on import"""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, device)
    return _model


@app.on_event("startup")
def warm_up():
    if Path(MODEL_PATH).exists():
        get_model()


# Endpoint to check the health of the API
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


# Endpoint to handle image uploads and return predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()

    try:
        return predict_bytes(get_model(), image_bytes, device)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
