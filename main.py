from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model

app = FastAPI()

# Google Drive file ID of your model
GDRIVE_FILE_ID = "18Fux2G1e8uuKFD5coZGj4T5OXKn26DcK"
MODEL_FILENAME = "face_shape_model_optimized.keras"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def download_model():
    print(f"Downloading model from Google Drive: {GDRIVE_URL}")
    gdown.download(GDRIVE_URL, MODEL_FILENAME, quiet=False)

# Download model if not present
if not os.path.exists(MODEL_FILENAME):
    download_model()

# Load the trained Keras model once when the server starts
model = load_model(MODEL_FILENAME)

# Labels based on your model's output
class_labels = ["diamond", "heart", "oval", "round", "square"]

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        return JSONResponse(status_code=400, content={"error": "Only PNG or JPG images are allowed"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(predictions)]
        return {"result": f"Detected face shape: {predicted_class}"}
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is not a valid image."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})
