from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Model loading (only once during startup)
MODEL_PATH = "model/nsfw-model.pt"
model = None
LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]

def load_model():
    global model
    try:
        model = torch.jit.load(MODEL_PATH)
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

def classify_image(image_tensor):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            result = {LABELS[i]: probabilities[i].item() * 100 for i in range(len(LABELS))}
        return result
    except Exception as e:
        logging.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/classify/", response_model=dict, responses={500: {"model": dict}})
async def classify_image_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_tensor = preprocess_image(contents)
        result = classify_image(image_tensor)
        return JSONResponse(content={"result": result})
    except HTTPException as e:
        raise e  # Re-raise HTTPException for FastAPI handling
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
