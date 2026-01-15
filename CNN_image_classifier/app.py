"""
Image Classification API
Cifar-10 Dataset

Transfer Learning with MobileNetV2
"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from PIL import Image
import io
import json
import tensorflow as tf
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(
    title="Image Classification API - Cifar-10 with MobileNetV2",
    description="An API for image classification using transfer learning with MobileNetV2 on the Cifar-10 dataset.",
    version="1.0.0"    
)

# Load model and class mappings
print("Loading model...")
model = load_model(
    "models/best_transfer_mobilenet.keras",
    custom_objects={'preprocess_input': preprocess_input}
)
print("Model loaded.")

with open("models/classes.json", "r") as f:
    classes = json.load(f)

# Convert string to integer keys
classes = {int(k): v for k, v in classes.items()}

# Response schema
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model: str
    classes: List[str]

def preprocess_image(image) -> np.ndarray:
    """
    Preprocess image for MobileNetV2 predictions

    1. Convert to rgb
    2. Resize to (96, 96)
    3. convert to numpy array
    4. add batch dimension
    5. preprocess for MobileNetV2
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((96, 96))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="ok",
        model="best_transfer_mobilenet",
        classes=list(classes.values())
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        model="MobileNetV2 Transfer Learning",
        classes=list(classes.values())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict image class
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image. {file.content_type} provided.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        precessed_image = preprocess_image(image)
        predictions = model.predict(precessed_image, verbose=0)
        probabilities = predictions[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = classes[predicted_index]
        confidence = float(probabilities[predicted_index])

        prob_dict = {
            classes[i]: float(prob) for i, prob in enumerate(probabilities)
        }

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict image classes for a batch of images
    """
    if len(files) > 32:
        raise HTTPException(
            status_code=400,
            detail="Max 32 images per batch allowed."
        )
    
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "Invalid file type. Please upload an image."
            })
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            precessed_image = preprocess_image(image)

            predictions = model.predict(precessed_image, verbose=0)
            probabilities = predictions[0]

            predicted_index = int(np.argmax(probabilities))

            results.append({
                "filename": file.filename,
                "predicted_class": classes[predicted_index],
                "confidence": float(probabilities[predicted_index])
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Error processing image: {str(e)}"
            }) 
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)