
# gatekeeper_app.py - Flask service for the Plant/Not-Plant Binary Classifier

import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np

# Import the model architecture and ngrok utility
from src.model import ResNet9 
from colab_utils import run_with_ngrok

# --- CONFIGURATION ---
MODEL_PATH = 'models/plant_gatekeeper_weights.pth'
CLASSES_PATH = 'gatekeeper_classes.txt'
NUM_CLASSES = 2  # 0: Plant, 1: Not_Plant
IMAGE_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5 # For classifying as Not_Plant

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# --- MODEL LOADING ---
CLASS_NAMES = []
model = None
MODEL_LOADED = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Loads the trained ResNet9 Gatekeeper model."""
    global CLASS_NAMES, model, MODEL_LOADED
    try:
        # Load class names
        with open(CLASSES_PATH, 'r') as f:
            CLASS_NAMES = [line.strip() for line in f.readlines()]
        
        # Initialize model architecture
        model = ResNet9(in_channels=3, num_classes=NUM_CLASSES)
        
        # Load weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        MODEL_LOADED = True
        print("✅ Gatekeeper Model loaded successfully.")
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH} or classes at {CLASSES_PATH}.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Call load model once on startup
load_model()


# --- TRANSFORMS for Inference ---
# Must match the training normalization
inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- API ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an image, classifies it as Plant (0) or Not_Plant (1),
    and returns the prediction.
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model is not yet loaded or initialized.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image type (e.g., JPEG, PNG)'}), 400

    try:
        # 1. Read and preprocess the image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = inference_transforms(image).unsqueeze(0).to(DEVICE) # Add batch dimension

        # 2. Make Prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the highest probability and its index
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item() * 100.0
            predicted_class = CLASS_NAMES[predicted_idx.item()]

        # 3. Decision Logic and Response
        is_plant = predicted_class == 'Plant'
        
        response = {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "is_plant": is_plant,
            "message": "Image is a plant, proceed to main disease classification." if is_plant else "Image is NOT a plant, reject further classification."
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "model_name": MODEL_PATH,
        "num_classes": NUM_CLASSES,
        "timestamp": datetime.now().isoformat()
    })

# --- ERROR HANDLERS ---
@app.errorhandler(413)
def too_large(e):
    """Handles file size exceeded error."""
    return jsonify({'error': 'File too large. Max 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handles not found errors."""
    return jsonify({'error': 'Endpoint not found. Check the URL.'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)