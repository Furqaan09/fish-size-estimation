import torch
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.predict import load_model, predict_bytes

app = FastAPI(title="Fish Segmentation API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("models/best_model.pth", device)


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
        return predict_bytes(model, image_bytes, device)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
