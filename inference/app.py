import io
import json

import torch

from PIL import Image

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

from src.artifact_downloader import (
    download_artifacts
)

from src.model_loader import (
    load_model
)

from src.image_transform import (
    create_transform
)

from src.predictor import (
    predict_image
)


# --------------------------------------------------
# Load Inference Config
# --------------------------------------------------

with open(
    "config.json",
    "r"
) as f:

    CONFIG = json.load(f)


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Startup
# --------------------------------------------------

print("Downloading Artifacts...")

ARTIFACT_DIR = download_artifacts(
    CONFIG
)

print("Loading Model...")

MODEL, CLASS_MAPPING, TRAIN_CONFIG = (
    load_model(
        ARTIFACT_DIR,
        DEVICE
    )
)

print("Creating Transform...")

TRANSFORM = create_transform(
    TRAIN_CONFIG
)

print("Inference Service Ready")


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Animal Classifier API"
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "running",
        "model_version":
        CONFIG["model"]["version"]
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = predict_image(
        image=image,
        model=MODEL,
        transform=TRANSFORM,
        class_mapping=CLASS_MAPPING,
        device=DEVICE
    )

    return result