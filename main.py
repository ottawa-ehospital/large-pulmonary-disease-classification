from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import io
from fastapi.middleware.cors import CORSMiddleware
import json
import gdown
import os
import cv2
import base64

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_id = os.getenv('GOOGLE_DRIVE_FILE_ID')
model_name = 'model_in_used.tflite'
gdown.download('https://drive.google.com/uc?id=' + file_id, model_name, quiet=False, fuzzy=True )


interpreter = tflite.Interpreter(model_path=model_name)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumothorax', 'Bacterial Pneumonia', 'Viral Pneumonia', 'Covid 19 Pneumonia', 'Tuberculosis']
max_limit = [0.208167, 0.217179, 0.229513, 0.121631, 0.122202, 0.096119, 1, 1, 1, 0.167872]
min_limit = [0.011248, 0.004896, 0.027108, 0.005163, 0.011129, 0.004800, 0, 0, 0, 0.002096]

class Prediction(BaseModel):
    predictions: dict
    heatmap: str

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128)).convert('L')
    image = np.array(image) / 255.0
    image = np.stack((image,) * 3, axis=-1) 
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image

def create_heatmap(feature_maps):
    heatmap = np.mean(feature_maps, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
    highlighted_img = cv2.addWeighted(gray_img, 1 - alpha, heatmap, alpha, 0)
    return highlighted_img

def log_transform(data):
    return np.log1p(data)    

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    layer_names = ['conv_pw_13_relu']
    intermediate_outputs = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(layer_names))]
    predictions = interpreter.get_tensor(output_details[len(layer_names)]['index'])
    
    predictions = log_transform(predictions).astype(float)

    predictions_dict = {class_labels[i]: ((predictions[0][i] - min(predictions[0][i], min_limit[i]))/max(predictions[0][i], max_limit[i])) for i in range(len(class_labels))}

    feature_maps = intermediate_outputs[0]

    # Generate the heatmap
    heatmap = create_heatmap(feature_maps[0])

    # Load the original image for visualization
    original_image = np.array(image.convert('RGB'))
    img = cv2.resize(original_image, (128, 128))

    superimposed_img = overlay_heatmap(img, heatmap, alpha=0.15, colormap=cv2.COLORMAP_JET)
    superimposed_img_resized = cv2.resize(superimposed_img, (original_image.shape[1], original_image.shape[0]))

    # Convert the superimposed image to base64 for response
    _, buffer = cv2.imencode('.png', superimposed_img_resized)
    heatmap_img_str = base64.b64encode(buffer).decode('utf-8')
    
    return JSONResponse(content={"predictions": predictions_dict, "heatmap": heatmap_img_str})
