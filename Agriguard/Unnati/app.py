# app.py - Enhanced Plant Disease Classifier with Gatekeeper Validation

import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import numpy as np

from src.model import ResNet9, GatekeeperResNet9
from disease_info import get_disease_info, DISEASE_INFO
from weather_utils import (
    get_weather_data, 
    get_location_name,
    analyze_weather_conditions, 
    get_weather_recommendations,
    get_seasonal_advice
)

# --- CONFIGURATION ---
MODEL_PATH = 'models/resnet9_plant_disease_weights.pth'
GATEKEEPER_MODEL_PATH = 'models/plant_gatekeeper_weights.pth'
CLASSES_PATH = 'plant_disease_classes.txt'
IMAGE_SIZE = 256
GATEKEEPER_IMAGE_SIZE = 128
CONFIDENCE_THRESHOLD = 50.0

# Gatekeeper threshold - only reject if NOT_PLANT confidence is above this
# Higher value = more lenient (fewer false rejections)
NOT_PLANT_REJECTION_THRESHOLD = 70.0  

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- MODEL LOADING ---
CLASS_NAMES = []
model = None
gatekeeper_model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global CLASS_NAMES, model, gatekeeper_model
    try:
        with open(CLASSES_PATH, 'r') as f:
            CLASS_NAMES = [line.strip() for line in f]
        NUM_CLASSES = len(CLASS_NAMES)
        
        model = ResNet9(in_channels=3, num_diseases=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ Disease classifier loaded successfully!")
        print(f"   - Classes: {NUM_CLASSES}")
        
        if os.path.exists(GATEKEEPER_MODEL_PATH):
            gatekeeper_model = GatekeeperResNet9(in_channels=3, num_classes=2)
            gatekeeper_model.load_state_dict(torch.load(GATEKEEPER_MODEL_PATH, map_location=DEVICE))
            gatekeeper_model.to(DEVICE)
            gatekeeper_model.eval()
            print("✅ Gatekeeper model loaded successfully!")
            print(f"   - Plant validation: ENABLED (rejection threshold: {NOT_PLANT_REJECTION_THRESHOLD}%)")
        else:
            print("⚠️ Gatekeeper model not found. Plant validation disabled.")
        
        print("   - Weather-aware recommendations: ENABLED")
        return True
    except Exception as e:
        print(f"❌ MODEL LOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Allow tests and lightweight runs to skip model loading by setting
# environment variable SKIP_MODEL_LOAD=1. This prevents heavy model loads
# when importing the app for unit tests.
if os.environ.get('SKIP_MODEL_LOAD', '0') == '1':
    print('WARNING: Skipping model loading (SKIP_MODEL_LOAD=1)')
    MODEL_LOADED = False
else:
    MODEL_LOADED = load_model()

# --- HELPER FUNCTIONS ---

def check_if_plant(image_bytes):
    """
    Use the gatekeeper model to check if image contains a plant.
    Returns: (is_plant, plant_confidence, not_plant_confidence, message)
    """
    global gatekeeper_model
    
    if gatekeeper_model is None:
        return True, 100.0, 0.0, "Gatekeeper not available"
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((GATEKEEPER_IMAGE_SIZE, GATEKEEPER_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = gatekeeper_model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Class 0 = Plant, Class 1 = Not_Plant
            plant_conf = probabilities[0][0].item() * 100
            not_plant_conf = probabilities[0][1].item() * 100
        
        print(f"   [Gatekeeper] Plant: {plant_conf:.1f}%, Not Plant: {not_plant_conf:.1f}%")
        
        # Only reject if we're HIGHLY confident it's NOT a plant
        # This prevents false rejections of actual plant images
        if not_plant_conf >= NOT_PLANT_REJECTION_THRESHOLD:
            return False, plant_conf, not_plant_conf, "Image does not appear to contain a plant leaf"
        else:
            return True, plant_conf, not_plant_conf, "Image accepted for analysis"
            
    except Exception as e:
        print(f"Gatekeeper error: {e}")
        return True, 0.0, 0.0, f"Validation error: {str(e)}"

def validate_image_basic(image_bytes):
    """Basic image validation."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        width, height = image.size
        if width < 32 or height < 32:
            return False, "Image too small (minimum 32x32 pixels required)"
        
        img_array = np.array(image)
        std_dev = np.std(img_array)
        if std_dev < 5:
            return False, "Image appears to be blank or completely uniform"
        
        mean_intensity = np.mean(img_array)
        if mean_intensity < 5:
            return False, "Image is too dark (appears to be all black)"
        if mean_intensity > 250:
            return False, "Image is too bright (appears to be all white)"
        
        return True, "Image validation passed"
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def preprocess_and_predict(image_bytes, language='en'):
    """Process image and return prediction with disease info."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        
    probabilities = F.softmax(outputs, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, min(3, len(CLASS_NAMES)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = CLASS_NAMES[idx.item()]
        confidence = prob.item() * 100
        disease_data = get_disease_info(class_name, language)
        predictions.append({
            "class_name": class_name,
            "confidence": confidence,
            "disease_info": disease_data
        })
    
    return predictions

def is_healthy(class_name):
    return "healthy" in class_name.lower()

def generate_treatment_calendar(disease_info, treatment, weather_conditions=None, class_name=None):
    """Generate a treatment schedule calendar."""
    calendar_events = []
    today = datetime.now()
    
    # To keep causation checks language-agnostic, try to inspect the
    # English "cause" in the master DISEASE_INFO (if available) using
    # class_name. If class_name isn't supplied or not present, fall back
    # to the localized cause text.
    cause_text = ""
    if class_name and class_name in DISEASE_INFO:
        cause_text = DISEASE_INFO[class_name].get("cause", {}).get("en", "").lower()
    else:
        cause_text = (disease_info.get("cause") or "").lower()

    is_fungal = "fungus" in cause_text or "fungal" in cause_text
    is_bacterial = "bacteria" in cause_text or "bacterial" in cause_text
    is_viral = "virus" in cause_text or "viral" in cause_text
    
    if is_viral:
        calendar_events.append({
            "day": 0, "date": today.strftime("%Y-%m-%d"),
            "title": "Remove Infected Plants",
            "description": "Immediately remove and destroy infected plants to prevent spread",
            "type": "urgent", "icon": "trash"
        })
        calendar_events.append({
            "day": 1, "date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "title": "Vector Control",
            "description": "Apply insecticides or organic controls for disease vectors",
            "type": "treatment", "icon": "spray-can"
        })
        for week in range(1, 5):
            calendar_events.append({
                "day": week * 7,
                "date": (today + timedelta(days=week*7)).strftime("%Y-%m-%d"),
                "title": f"Week {week} Monitoring",
                "description": "Check plants for symptoms and maintain vector control",
                "type": "monitoring", "icon": "search"
            })
    
    elif is_fungal or is_bacterial:
        calendar_events.append({
            "day": 0, "date": today.strftime("%Y-%m-%d"),
            "title": "First Treatment Application",
            "description": treatment["organic_remedies"][0] if treatment["organic_remedies"] else "Apply recommended treatment",
            "type": "treatment", "icon": "spray-can"
        })
        
        for i, day in enumerate([7, 14, 21, 28]):
            calendar_events.append({
                "day": day,
                "date": (today + timedelta(days=day)).strftime("%Y-%m-%d"),
                "title": f"Treatment {i+2}",
                "description": "Apply treatment as per schedule.",
                "type": "treatment", "icon": "spray-can"
            })
        
        for day in [3, 10, 17, 24, 31]:
            calendar_events.append({
                "day": day,
                "date": (today + timedelta(days=day)).strftime("%Y-%m-%d"),
                "title": "Monitor Progress",
                "description": "Inspect plants for improvement or worsening symptoms",
                "type": "monitoring", "icon": "search"
            })
    
    calendar_events.append({
        "day": 35, "date": (today + timedelta(days=35)).strftime("%Y-%m-%d"),
        "title": "Prevention Review",
        "description": disease_info.get("prevention", "Review preventive measures"),
        "type": "prevention", "icon": "shield-alt"
    })
    
    if weather_conditions:
        if weather_conditions.get("rain_expected"):
            calendar_events.append({
                "day": 2, "date": (today + timedelta(days=2)).strftime("%Y-%m-%d"),
                "title": "⚠️ Rain Expected",
                "description": "Apply systemic treatments before rain.",
                "type": "weather-alert", "icon": "cloud-rain"
            })
        if weather_conditions.get("is_humid"):
            calendar_events.append({
                "day": 5, "date": (today + timedelta(days=5)).strftime("%Y-%m-%d"),
                "title": "High Humidity Alert",
                "description": "Increase monitoring frequency.",
                "type": "weather-alert", "icon": "tint"
            })
    
    calendar_events.sort(key=lambda x: x["day"])
    return calendar_events

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    weather_data = get_weather_data(lat, lon)
    if not weather_data:
        return jsonify({'error': 'Could not fetch weather data'}), 500
    
    location_name = get_location_name(lat, lon)
    conditions = analyze_weather_conditions(weather_data)
    
    return jsonify({
        'success': True,
        'location': {'name': location_name, 'latitude': lat, 'longitude': lon},
        'current_weather': {
            'temperature': conditions['temperature'],
            'humidity': conditions['humidity'],
            'wind_speed': conditions['wind_speed'],
            'description': conditions['weather_description'],
            'precipitation': conditions['precipitation']
        },
        'conditions': {
            'is_rainy': conditions['is_rainy'],
            'is_humid': conditions['is_humid'],
            'is_hot': conditions['is_hot'],
            'is_cold': conditions['is_cold'],
            'is_windy': conditions['is_windy'],
            'rain_expected': conditions['rain_expected']
        },
        'seasonal_advice': get_seasonal_advice(conditions)
    })

@app.route('/api/diseases', methods=['GET'])
def list_diseases():
    # Get language parameter from query string (default to 'en')
    language = request.args.get('language', 'en').lower()
    # Accept any 2-letter language code, default to 'en' if invalid format
    if not language or len(language) != 2 or not language.isalpha():
        language = 'en'
    
    diseases = []
    for class_name, info in DISEASE_INFO.items():
        disease_info = get_disease_info(class_name, language)
        diseases.append({
            "class_name": class_name,
            "disease": disease_info["disease"],
            "plant": disease_info["plant"]
        })
    return jsonify({"diseases": diseases, "count": len(diseases)})

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get language parameter from query string or form data (default to 'en')
    language = request.args.get('language', request.form.get('language', 'en')).lower()
    # Accept any 2-letter language code, default to 'en' if invalid format
    if not language or len(language) != 2 or not language.isalpha():
        language = 'en'
    
    lat = request.form.get('latitude', type=float)
    lon = request.form.get('longitude', type=float)
    generate_calendar = request.form.get('generate_calendar', 'false').lower() == 'true'
    
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed)}'}), 400
    
    try:
        img_bytes = file.read()
        
        # Step 1: Basic validation
        is_valid, validation_reason = validate_image_basic(img_bytes)
        if not is_valid:
            return jsonify({
                'success': False, 'error': 'Invalid Image', 'validation_failed': True,
                'reason': validation_reason,
                'suggestion': 'Please upload a clear image of a plant leaf.'
            }), 400
        
        # Step 2: Check if plant using gatekeeper
        is_plant, plant_conf, not_plant_conf, plant_message = check_if_plant(img_bytes)
        
        if not is_plant:
            return jsonify({
                'success': False, 'error': 'Not a Plant Image', 'validation_failed': True,
                'reason': f'{plant_message} (Confidence: {not_plant_conf:.1f}%)',
                'suggestion': 'Please upload an image of a plant leaf. This appears to be something else.',
                'detection_details': {
                    'is_plant': False, 
                    'plant_confidence': f'{plant_conf:.2f}%',
                    'not_plant_confidence': f'{not_plant_conf:.2f}%'
                }
            }), 400
        
        # Step 3: Disease classification
        predictions = preprocess_and_predict(img_bytes, language)
        
        top_pred = predictions[0]
        confidence = top_pred["confidence"]
        disease_info = top_pred["disease_info"]
        plant_is_healthy = is_healthy(top_pred["class_name"])
        
        # Weather recommendations
        weather_recommendations = None
        weather_info = None
        location_info = None
        weather_conditions = None
        
        if lat is not None and lon is not None:
            weather_data = get_weather_data(lat, lon)
            if weather_data:
                weather_conditions = analyze_weather_conditions(weather_data)
                weather_recommendations = get_weather_recommendations(weather_conditions, plant_is_healthy)
                location_info = {'name': get_location_name(lat, lon), 'latitude': lat, 'longitude': lon}
                weather_info = {
                    'temperature': f"{weather_conditions['temperature']}°C",
                    'humidity': f"{weather_conditions['humidity']}%",
                    'wind_speed': f"{weather_conditions['wind_speed']} km/h",
                    'description': weather_conditions['weather_description'],
                    'is_rainy': weather_conditions['is_rainy'],
                    'rain_expected': weather_conditions['rain_expected'],
                    'forecast_rain': f"{weather_conditions['forecast_rain_mm']:.1f}mm in 3 days"
                }
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "validation": {
                "is_plant": True, 
                "plant_confidence": f"{plant_conf:.2f}%",
                "not_plant_confidence": f"{not_plant_conf:.2f}%"
            },
            "prediction": {
                "class_name": top_pred["class_name"],
                "confidence": f"{confidence:.2f}%",
                "confidence_value": confidence,
                "is_healthy": plant_is_healthy,
                "is_reliable": confidence >= CONFIDENCE_THRESHOLD
            },
            "disease_info": {
                "name": disease_info["disease"],
                "plant": disease_info["plant"],
                "cause": disease_info["cause"],
                "symptoms": disease_info["symptoms"]
            },
            "treatment": {
                "organic_remedies": disease_info["organic_remedies"],
                "chemical_remedies": disease_info["chemical_remedies"],
                "prevention": disease_info["prevention"]
            },
            "alternative_predictions": [
                {"class_name": p["class_name"], "confidence": f"{p['confidence']:.2f}%", "disease": p["disease_info"]["disease"]}
                for p in predictions[1:]
            ]
        }
        
        if weather_info:
            response["location"] = location_info
            response["weather"] = weather_info
            response["weather_recommendations"] = weather_recommendations
        
        if generate_calendar and not plant_is_healthy:
            # pass top_pred class_name so the calendar generator can check
            # cause in English (language-independent checks)
            response["treatment_calendar"] = generate_treatment_calendar(
                disease_info,
                response["treatment"],
                weather_conditions,
                class_name=top_pred["class_name"]
            )
        
        if confidence < CONFIDENCE_THRESHOLD:
            response["warning"] = "Low confidence. Retake photo with better lighting."
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "gatekeeper_loaded": gatekeeper_model is not None,
        "rejection_threshold": NOT_PLANT_REJECTION_THRESHOLD,
        "weather_enabled": True,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found. Check the URL.'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)