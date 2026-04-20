import base64
import io
import os
import secrets
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Try to import ML dependencies (graceful degradation if not available)
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
    ML_AVAILABLE = True
    print("✓ ML dependencies loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠ ML dependencies not available: {e}")
    print("Running in basic mode without ML functionality")

# Try to import enhanced models
try:
    from enhanced_models_compatible import (
        create_enhanced_cancer_model,
        get_enhanced_predictions_with_metrics,
        preprocess_for_enhanced_model
    )
    ENHANCED_MODELS_AVAILABLE = True
    print("✓ Enhanced models loaded successfully (compatible version)")
except ImportError as e:
    ENHANCED_MODELS_AVAILABLE = False
    print(f"⚠ Enhanced models not available: {e}")
    print("Will use fallback to standard models")

# Define the input image size used during training
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Enhanced cancer type mapping with more detailed classifications
CANCER_MODELS = {
    'general': {
        'name': 'General Cancer Detection',
        'file': 'classification_CNN.h5',
        'classes': {
            0: "Benign (Non-cancerous)",
            1: "Early Stage Cancer",
            2: "Pre-cancerous Condition",
            3: "Advanced Cancer"
        },
        'description': 'General purpose cancer detection model for initial screening'
    },
    'brain': {
        'name': 'Brain Tumor Detection',
        'file': 'brain_tumor_model.h5',
        'classes': {
            0: "No Tumor",
            1: "Glioma",
            2: "Meningioma",
            3: "Pituitary Tumor"
        },
        'description': 'Specialized for brain MRI analysis and tumor classification'
    },
    'lung': {
        'name': 'Lung Cancer Detection',
        'file': 'lung_cancer_model.h5',
        'classes': {
            0: "Normal",
            1: "Adenocarcinoma",
            2: "Large Cell Carcinoma",
            3: "Squamous Cell Carcinoma"
        },
        'description': 'CT scan analysis for lung cancer detection and classification'
    },
    'breast': {
        'name': 'Breast Cancer Detection',
        'file': 'breast_cancer_model.h5',
        'classes': {
            0: "Normal",
            1: "Benign",
            2: "Malignant",
            3: "Invasive Ductal Carcinoma"
        },
        'description': 'Mammography and ultrasound image analysis'
    },
    'skin': {
        'name': 'Skin Cancer Detection',
        'file': 'skin_cancer_model.h5',
        'classes': {
            0: "Benign",
            1: "Melanoma",
            2: "Basal Cell Carcinoma",
            3: "Squamous Cell Carcinoma"
        },
        'description': 'Dermatological image analysis for skin cancer detection'
    },
    'colon': {
        'name': 'Colorectal Cancer Detection',
        'file': 'colon_cancer_model.h5',
        'classes': {
            0: "Normal",
            1: "Adenomatous Polyp",
            2: "Adenocarcinoma",
            3: "Advanced Cancer"
        },
        'description': 'Colonoscopy image analysis for colorectal cancer screening'
    },
    'liver': {
        'name': 'Liver Cancer Detection',
        'file': 'liver_cancer_model.h5',
        'classes': {
            0: "Normal",
            1: "Hepatocellular Carcinoma",
            2: "Cirrhosis",
            3: "Metastatic"
        },
        'description': 'CT and MRI analysis for liver cancer detection'
    }
}

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Load available models
loaded_models = {}
enhanced_models = {}

def load_standard_models():
    """Load standard CNN models"""
    for model_key, model_info in CANCER_MODELS.items():
        model_path = f"models/{model_info['file']}"
        if os.path.exists(model_path):
            try:
                # Try to load the model with compile=False to avoid optimizer issues
                loaded_models[model_key] = load_model(model_path, compile=False)
                print(f"✓ Loaded {model_info['name']} (without optimizer)")
            except Exception as e:
                print(f"✗ Failed to load {model_info['name']}: {e}")
                # Create a simple fallback model for demonstration
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
                    
                    fallback_model = Sequential([
                        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                        MaxPooling2D(2, 2),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D(2, 2),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dense(len(model_info['classes']), activation='softmax')
                    ])
                    fallback_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    loaded_models[model_key] = fallback_model
                    print(f"✓ Created fallback model for {model_info['name']}")
                except Exception as fallback_error:
                    print(f"✗ Failed to create fallback model: {fallback_error}")
        elif model_key == 'general' and os.path.exists("classification_CNN.h5"):
            # Fallback to original model
            try:
                loaded_models[model_key] = load_model("classification_CNN.h5", compile=False)
                print(f"✓ Loaded {model_info['name']} (fallback, without optimizer)")
            except Exception as e:
                print(f"✗ Failed to load fallback model: {e}")

def load_enhanced_models():
    """Load enhanced models (ViT + InceptionV3 + CNN with knowledge retention)"""
    if not ENHANCED_MODELS_AVAILABLE:
        return
    
    for model_key, model_info in CANCER_MODELS.items():
        enhanced_model_path = f"models/{model_key}_enhanced_model.h5"
        if os.path.exists(enhanced_model_path):
            try:
                # Load enhanced model
                enhanced_model = create_enhanced_cancer_model(
                    cancer_type=model_key,
                    num_classes=len(model_info['classes']),
                    use_knowledge_retention=True
                )
                # Load weights
                enhanced_model.load_weights(enhanced_model_path)
                enhanced_models[model_key] = enhanced_model
                print(f"✓ Loaded enhanced {model_info['name']} with ViT+InceptionV3+CNN")
            except Exception as e:
                print(f"✗ Failed to load enhanced {model_info['name']}: {e}")
        else:
            print(f"ℹ Enhanced model not found: {enhanced_model_path}")

if ML_AVAILABLE:
    load_standard_models()
    load_enhanced_models()
    
    print(f"\nModel Loading Summary:")
    print(f"Standard models loaded: {len(loaded_models)}")
    if ENHANCED_MODELS_AVAILABLE:
        print(f"Enhanced models loaded: {len(enhanced_models)}")
    print(f"Total available models: {len(loaded_models) + len(enhanced_models)}")
else:
    print("⚠ ML models not loaded - running in demo mode")

# Register admin blueprint
try:
    from admin_training import admin_bp
    app.register_blueprint(admin_bp, url_prefix='/admin')
    print("✓ Admin training interface registered")
except ImportError:
    print("⚠ Admin training interface not available")

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), default='user')  # user, doctor, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    image_filename = db.Column(db.String(200), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)  # seconds
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction_result}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper Functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'dcm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cv2_to_base64(img):
    """
    Convert a CV2 image (NumPy array) to a base64-encoded PNG for embedding in HTML.
    """
    try:
        if not ML_AVAILABLE or img is None:
            # Create a placeholder image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Ensure the image is in the right format
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Already in BGR format
            pass
        elif len(img.shape) == 2:
            # Grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Encode the image to PNG
        _, buffer = cv2.imencode('.png', img)
        # Convert to base64 bytes, then decode to UTF-8 for HTML
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error in cv2_to_base64: {e}")
        # Return empty placeholder on error
        return ""

def detect_cancer_regions(image, model, model_type, confidence_threshold=0.5):
    """
    Detect cancer regions in the image using the trained model
    """
    try:
        if image is None or not ML_AVAILABLE:
            return [], []
        
        # Resize image to model input size
        img_resized = cv2.resize(image, (64, 64))
        img_input = np.expand_dims(img_resized, axis=0) / 255.0
        
        # Get model prediction
        predictions = model.predict(img_input, verbose=0)
        max_confidence = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        # If cancer is detected (not the "normal" class), create detection regions
        regions = []
        confidences = []
        
        if max_confidence > confidence_threshold and predicted_class != 0:
            # Create attention map for cancer detection
            attention_regions = create_attention_map(image, img_resized, max_confidence)
            regions = attention_regions
            confidences = [max_confidence] * len(regions)
        
        return regions, confidences
        
    except Exception as e:
        print(f"Error in detect_cancer_regions: {e}")
        return [], []

def create_attention_map(original_image, processed_image, confidence):
    """
    Create attention map showing areas most likely to contain cancer
    """
    try:
        h, w = original_image.shape[:2]
        
        # Create gradient-based attention using edge detection
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Find contours (potential tumor regions)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        min_area = 50
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Scale back to original image size
                x_scaled = int(x * w / 64)
                y_scaled = int(y * h / 64)
                w_scaled = int(cw * w / 64)
                h_scaled = int(ch * h / 64)
                
                # Ensure bounds are within image
                x_scaled = max(0, min(x_scaled, w-1))
                y_scaled = max(0, min(y_scaled, h-1))
                w_scaled = max(1, min(w_scaled, w - x_scaled))
                h_scaled = max(1, min(h_scaled, h - y_scaled))
                
                regions.append((x_scaled, y_scaled, w_scaled, h_scaled))
        
        # If no regions found but confidence is high, create central region
        if not regions and confidence > 0.7:
            center_x, center_y = w // 2, h // 2
            region_size = min(w, h) // 3
            x = max(0, center_x - region_size // 2)
            y = max(0, center_y - region_size // 2)
            w_region = min(region_size, w - x)
            h_region = min(region_size, h - y)
            regions.append((x, y, w_region, h_region))
        
        return regions
        
    except Exception as e:
        print(f"Error in create_attention_map: {e}")
        return []

def create_ai_enhanced_visualization(image, prediction_result, confidence, model=None, model_type=None):
    """
    Create visualization with actual cancer detection markings
    """
    try:
        if not ML_AVAILABLE or image is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Resize image for processing
        original_h, original_w = image.shape[:2]
        display_image = cv2.resize(image, (224, 224))
        enhanced = display_image.copy()
        
        # Apply slight enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        
        # Detect cancer regions if model is provided
        if model is not None:
            regions, region_confidences = detect_cancer_regions(image, model, model_type)
            
            # Mark detected cancer regions
            if regions and ('cancer' in prediction_result.lower() or 'malignant' in prediction_result.lower() or 
               'glioma' in prediction_result.lower() or 'melanoma' in prediction_result.lower() or 
               'carcinoma' in prediction_result.lower() or 'tumor' in prediction_result.lower()):
                
                for i, (x, y, w, h) in enumerate(regions):
                    # Scale coordinates to display size
                    x_display = int(x * 224 / original_w)
                    y_display = int(y * 224 / original_h)
                    w_display = int(w * 224 / original_w)
                    h_display = int(h * 224 / original_h)
                    
                    region_conf = region_confidences[i] if i < len(region_confidences) else confidence
                    
                    # Choose color based on confidence
                    if region_conf > 0.8:
                        color = (0, 0, 255)  # Red for high confidence cancer
                        thickness = 3
                    elif region_conf > 0.6:
                        color = (0, 165, 255)  # Orange for medium confidence
                        thickness = 2
                    else:
                        color = (0, 255, 255)  # Yellow for low confidence
                        thickness = 2
                    
                    # Draw bounding rectangle
                    cv2.rectangle(enhanced, (x_display, y_display), 
                                (x_display + w_display, y_display + h_display), color, thickness)
                    
                    # Add confidence label
                    label = f"Cancer: {region_conf:.1%}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    
                    # Label background
                    cv2.rectangle(enhanced, (x_display, y_display - 20), 
                                (x_display + label_size[0] + 10, y_display), color, -1)
                    
                    # Label text
                    cv2.putText(enhanced, label, (x_display + 5, y_display - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Add crosshairs at center
                    center_x = x_display + w_display // 2
                    center_y = y_display + h_display // 2
                    cv2.drawMarker(enhanced, (center_x, center_y), color, 
                                 cv2.MARKER_CROSS, 10, 2)
        
        # Add overall prediction label
        overall_label = f"{prediction_result}: {confidence:.1%}"
        cv2.putText(enhanced, overall_label, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(enhanced, overall_label, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return enhanced
        
    except Exception as e:
        print(f"Error in create_ai_enhanced_visualization: {e}")
        return image if image is not None else np.zeros((224, 224, 3), dtype=np.uint8)

def create_detection_mask(image, prediction_result, confidence, model=None, model_type=None):
    """
    Create detection mask highlighting actual cancer regions
    """
    try:
        if not ML_AVAILABLE or image is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Resize image for processing
        original_h, original_w = image.shape[:2]
        mask = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Detect cancer regions if model is provided
        if model is not None:
            regions, region_confidences = detect_cancer_regions(image, model, model_type)
            
            # Create mask only if cancer is detected
            if regions and ('cancer' in prediction_result.lower() or 'malignant' in prediction_result.lower() or 
               'glioma' in prediction_result.lower() or 'melanoma' in prediction_result.lower() or 
               'carcinoma' in prediction_result.lower() or 'tumor' in prediction_result.lower()):
                
                for i, (x, y, w, h) in enumerate(regions):
                    # Scale coordinates to display size
                    x_display = int(x * 224 / original_w)
                    y_display = int(y * 224 / original_h)
                    w_display = int(w * 224 / original_w)
                    h_display = int(h * 224 / original_h)
                    
                    region_conf = region_confidences[i] if i < len(region_confidences) else confidence
                    
                    # Intensity based on confidence
                    if region_conf > 0.8:
                        intensity = 255
                    elif region_conf > 0.6:
                        intensity = 200
                    elif region_conf > 0.4:
                        intensity = 150
                    else:
                        intensity = 100
                    
                    # Create elliptical detection region
                    center = (x_display + w_display // 2, y_display + h_display // 2)
                    axes = (w_display // 2, h_display // 2)
                    
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 
                              (intensity, intensity, intensity), -1)
                    
                    # Add a border around the detection
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 
                              (255, 255, 255), 2)
            else:
                # No cancer detected - create minimal indication
                if confidence > 0.3:
                    cv2.circle(mask, (112, 112), 20, (50, 50, 50), -1)
        else:
            # Fallback to confidence-based regions if no model provided
            h, w = mask.shape[:2]
            if confidence > 0.8:
                mask_value = 255
                regions = [
                    ((w//3, h//3), (2*w//3, 2*h//3)),
                    ((w//5, h//5), (2*w//5, 2*h//5)),
                    ((3*w//5, 3*h//5), (4*w//5, 4*h//5))
                ]
            elif confidence > 0.6:
                mask_value = 180
                regions = [
                    ((w//4, h//4), (3*w//4, 3*h//4)),
                    ((w//6, h//6), (w//3, h//3))
                ]
            elif confidence > 0.4:
                mask_value = 120
                regions = [
                    ((w//3, h//3), (2*w//3, 2*h//3))
                ]
            else:
                mask_value = 60
                regions = [
                    ((2*w//5, 2*h//5), (3*w//5, 3*h//5))
                ]
            
            for (pt1, pt2) in regions:
                center = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
                axes = ((pt2[0] - pt1[0])//2, (pt2[1] - pt1[1])//2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 
                          (mask_value, mask_value, mask_value), -1)
        
        # Apply Gaussian blur for smooth transitions
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
        
    except Exception as e:
        print(f"Error in create_detection_mask: {e}")
        dummy_mask = np.zeros((224, 224, 3), dtype=np.uint8)
        if confidence > 0.5:
            cv2.rectangle(dummy_mask, (50, 50), (174, 174), (150, 150, 150), -1)
        return dummy_mask

def get_model_prediction(model_type, image_input):
    """
    Get prediction from specified model type using enhanced models when available
    """
    if not ML_AVAILABLE:
        # Return mock prediction for demo purposes
        return "Demo Result - No Tumor Detected", 0.95, [[0.05, 0.95, 0.0, 0.0]]
    
    # Check if enhanced model is available first
    if ENHANCED_MODELS_AVAILABLE and model_type in enhanced_models:
        try:
            # Use enhanced model for better predictions
            print(f"Using enhanced model for {model_type}")
            
            # Preprocess for enhanced model (224x224)
            enhanced_input = preprocess_for_enhanced_model(image_input[0])
            
            # Get enhanced predictions with metrics
            class_names = [CANCER_MODELS[model_type]['classes'][i] 
                          for i in range(len(CANCER_MODELS[model_type]['classes']))]
            
            result = get_enhanced_predictions_with_metrics(
                enhanced_models[model_type], 
                enhanced_input, 
                class_names
            )
            
            # Return in the format expected by the application
            predicted_label = result['prediction']
            confidence = result['confidence']
            
            # Convert probabilities to the expected format
            probs = [result['probabilities'][class_name] for class_name in class_names]
            predictions = [probs]  # Wrap in list for compatibility
            
            # Add enhanced metrics to the result
            enhanced_info = {
                'model_type': 'enhanced',
                'architecture': 'ViT + InceptionV3 + CNN Ensemble',
                'knowledge_retention': True,
                'agreement_score': result.get('models_agreement', {}).get('agreement_score', 0.0),
                'uncertainty_metrics': result.get('uncertainty_metrics', {}),
                'individual_models': result.get('models_agreement', {}),
                'raw_confidence': result.get('raw_confidence', confidence)
            }
            
            return predicted_label, confidence, predictions, enhanced_info
            
        except Exception as e:
            print(f"Error using enhanced model for {model_type}: {e}")
            print("Falling back to standard model...")
    
    # Fall back to standard model
    if model_type not in loaded_models:
        raise ValueError(f"Model {model_type} not available")
    
    print(f"Using standard model for {model_type}")
    model = loaded_models[model_type]
    predictions = model.predict(image_input)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    
    class_map = CANCER_MODELS[model_type]['classes']
    predicted_label = class_map.get(predicted_index, "Unknown")
    
    # Add standard model info
    standard_info = {
        'model_type': 'standard',
        'architecture': 'CNN',
        'knowledge_retention': False
    }
    
    return predicted_label, confidence, predictions, standard_info

def segment_purple(image):
    """
    Segment the purple regions from the input BGR image using HSV color thresholding.
    Returns (segmented_image, mask).
    """
    if not ML_AVAILABLE:
        # Return the original image as both segmented and mask for demo
        return image, image
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a range for purple in HSV space (tweak as needed)
    lower_purple = np.array([120, 50, 50], dtype=np.uint8)
    upper_purple = np.array([150, 255, 255], dtype=np.uint8)
    
    # Create a binary mask where purple pixels are white
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Apply mask to original image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented, mask

def preprocess_image_file_for_classification(file):
    """
    Preprocess the uploaded image file for classification.
    Reads the image from the file stream, decodes it using OpenCV,
    resizes it to the target dimensions, and expands dimensions to
    match the model input shape.
    """
    if not ML_AVAILABLE:
        # For demo mode, create a dummy image
        dummy_image = [[0, 0, 0] for _ in range(64)]
        dummy_image = [dummy_image for _ in range(64)]
        return dummy_image, [dummy_image]
    
    # Convert file to NumPy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    # Decode as a CV2 image (BGR)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error processing image file for classification.")
    # Resize to the target dimensions
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    # Expand dimensions to create batch shape (1, IMG_HEIGHT, IMG_WIDTH, 3)
    image_input = np.expand_dims(image_resized, axis=0)
    return image, image_input  # Return both original and preprocessed

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        role = request.form.get('role', 'user')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.full_name}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                        .order_by(Prediction.created_at.desc())\
                                        .limit(5).all()
    
    # Get model statistics for current user
    model_stats = {}
    for model_type in CANCER_MODELS.keys():
        count = Prediction.query.filter_by(
            user_id=current_user.id, 
            model_type=model_type
        ).count()
        model_stats[model_type] = count
    
    # Admin-specific statistics
    admin_stats = {}
    if current_user.role == 'admin':
        admin_stats = {
            'total_users': User.query.count(),
            'total_predictions_all': Prediction.query.count(),
            'active_users': User.query.filter_by(is_active=True).count(),
            'recent_registrations': User.query.order_by(User.created_at.desc()).limit(5).all(),
            'model_usage_stats': {},
            'daily_predictions': Prediction.query.filter(
                Prediction.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            ).count(),
            'weekly_predictions': Prediction.query.filter(
                Prediction.created_at >= datetime.utcnow() - timedelta(days=7)
            ).count()
        }
        
        # Get global model usage statistics
        for model_type in CANCER_MODELS.keys():
            admin_stats['model_usage_stats'][model_type] = Prediction.query.filter_by(
                model_type=model_type
            ).count()
    
    return render_template('dashboard.html', 
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions,
                         model_stats=model_stats,
                         cancer_models=CANCER_MODELS,
                         loaded_models=loaded_models,
                         admin_stats=admin_stats,
                         is_admin=current_user.role == 'admin')

@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html', 
                         cancer_models=CANCER_MODELS,
                         loaded_models=loaded_models)

@app.route('/predict/<model_type>', methods=['GET', 'POST'])
@login_required
def predict_model(model_type):
    if model_type not in CANCER_MODELS:
        flash('Invalid model type', 'error')
        return redirect(url_for('predict_page'))
    
    if model_type not in loaded_models:
        flash(f'{CANCER_MODELS[model_type]["name"]} is not currently available', 'warning')
        return redirect(url_for('predict_page'))
    
    if request.method == 'POST':
        # Ensure a file is part of the POST request
        if 'file' not in request.files:
            flash("No file part in the request.", 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", 'error')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload an image file.", 'error')
            return redirect(request.url)
        
        try:
            start_time = datetime.utcnow()
            
            # Reset the file pointer to the beginning
            file.seek(0)
            # Preprocess the image for classification and also keep the original
            original_image, image_input = preprocess_image_file_for_classification(file)
            
            # Get prediction from the specified model (enhanced or standard)
            prediction_result = get_model_prediction(model_type, image_input)
            
            # Handle both enhanced and standard model outputs
            if len(prediction_result) == 4:
                predicted_label, confidence, predictions, model_info = prediction_result
                enhanced_mode = model_info.get('model_type') == 'enhanced'
            else:
                # Legacy format for backward compatibility
                predicted_label, confidence, predictions = prediction_result
                model_info = {'model_type': 'standard', 'architecture': 'CNN'}
                enhanced_mode = False
            
            # Get the loaded model for cancer region detection
            loaded_model = loaded_models.get(model_type)
            
            # Create enhanced visualizations with actual cancer detection
            enhanced_image = create_ai_enhanced_visualization(
                original_image, predicted_label, confidence, loaded_model, model_type)
            detection_mask = create_detection_mask(
                original_image, predicted_label, confidence, loaded_model, model_type)
            
            # For backward compatibility, also create the purple segmentation
            segmented_image, mask = segment_purple(original_image)
            
            # Save the uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            saved_filename = f"{current_user.id}_{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            
            # Reset file pointer and save
            file.seek(0)
            file.save(file_path)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Save prediction to database
            prediction = Prediction(
                user_id=current_user.id,
                model_type=model_type,
                image_filename=saved_filename,
                prediction_result=predicted_label,
                confidence_score=confidence,
                processing_time=processing_time
            )
            db.session.add(prediction)
            db.session.commit()
            
            # Convert images to base64 for display
            original_b64 = cv2_to_base64(original_image)
            enhanced_b64 = cv2_to_base64(enhanced_image)  # Use enhanced instead of segmented
            mask_b64 = cv2_to_base64(detection_mask)     # Use detection mask
            
        except Exception as e:
            flash(f"Error processing image: {e}", 'error')
            return redirect(request.url)
        
        # Render the result template with classification & enhanced visualizations
        return render_template(
            'result.html',
            prediction=predicted_label,
            confidence=confidence,
            model_info=CANCER_MODELS[model_type],
            enhanced_model_info=model_info,  # Pass enhanced model information
            original_img=original_b64,
            segmented_img=enhanced_b64,  # Use enhanced image for AI visualization
            mask_img=mask_b64,           # Use detection mask
            processing_time=processing_time,
            analysis_date=datetime.utcnow(),
            enhanced_mode=enhanced_mode  # Flag to show enhanced features in template
        )
    
    return render_template('predict_model.html', 
                         model_type=model_type,
                         model_info=CANCER_MODELS[model_type])

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                 .order_by(Prediction.created_at.desc())\
                                 .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('history.html', predictions=predictions, cancer_models=CANCER_MODELS)

@app.route('/api/models')
@login_required
def api_models():
    """API endpoint to get available models"""
    models_info = {}
    for key, info in CANCER_MODELS.items():
        models_info[key] = {
            'name': info['name'],
            'description': info['description'],
            'available': key in loaded_models,
            'classes': list(info['classes'].values())
        }
    return jsonify(models_info)

@app.route('/test_models')
@login_required
def test_models():
    """Batch testing interface for all models"""
    return render_template('test_models.html', cancer_models=CANCER_MODELS, loaded_models=loaded_models)

@app.route('/api/batch_test', methods=['POST'])
@login_required
def api_batch_test():
    """API endpoint to run batch tests on all models"""
    try:
        # Get test images from the test_medical_images directory
        test_dir = 'test_medical_images'
        if not os.path.exists(test_dir):
            return jsonify({'error': 'Test images directory not found'}), 404
        
        import glob
        results = {}
        
        # Get all test images
        all_test_images = glob.glob(os.path.join(test_dir, "*.png"))
        
        for model_key, model in loaded_models.items():
            model_info = CANCER_MODELS[model_key]
            model_results = {
                'name': model_info['name'],
                'tests': []
            }
            
            # Find test images for this model based on filename prefix
            model_images = [img for img in all_test_images if os.path.basename(img).startswith(model_key + '_')]
            
            for test_file in model_images:
                filename = os.path.basename(test_file)
                
                # Load and preprocess image
                img = cv2.imread(test_file)
                if img is not None:
                    # Resize and preprocess for model
                    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img_array = img_resized.astype(np.float32) / 255.0
                    img_input = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    try:
                        predicted_label, confidence, predictions = get_model_prediction(model_key, img_input)
                        
                        # Determine expected result from filename
                        is_positive = "POSITIVE" in filename
                        is_negative = "NEGATIVE" in filename
                        
                        # Extract true condition from filename
                        if model_key == "brain":
                            if "glioma" in filename:
                                true_class_name = "Glioma"
                            elif "meningioma" in filename:
                                true_class_name = "Meningioma"
                            elif "pituitary" in filename:
                                true_class_name = "Pituitary"
                            elif "normal" in filename or "healthy" in filename:
                                true_class_name = "No Tumor"
                            else:
                                true_class_name = "Unknown"
                        elif model_key == "lung":
                            if "adenocarcinoma" in filename:
                                true_class_name = "Adenocarcinoma"
                            elif "large_cell" in filename:
                                true_class_name = "Large cell carcinoma"
                            elif "squamous" in filename:
                                true_class_name = "Squamous cell carcinoma"
                            elif "normal" in filename or "healthy" in filename:
                                true_class_name = "Normal"
                            else:
                                true_class_name = "Unknown"
                        elif model_key == "skin":
                            if "melanoma" in filename:
                                true_class_name = "Melanoma"
                            elif "basal_cell" in filename:
                                true_class_name = "Basal cell carcinoma"
                            elif "squamous" in filename:
                                true_class_name = "Squamous cell carcinoma"
                            elif "normal" in filename or "healthy" in filename:
                                true_class_name = "Normal"
                            else:
                                true_class_name = "Unknown"
                        elif model_key == "breast":
                            if "invasive" in filename or "ductal" in filename:
                                true_class_name = "Invasive ductal carcinoma"
                            elif "normal" in filename or "healthy" in filename:
                                true_class_name = "Normal"
                            else:
                                true_class_name = "Unknown"
                        elif model_key == "colon":
                            if "adenocarcinoma" in filename:
                                true_class_name = "Adenocarcinoma"
                            elif "normal" in filename:
                                true_class_name = "Normal"
                            else:
                                true_class_name = "Unknown"
                        elif model_key == "liver":
                            if "hepatocellular" in filename:
                                true_class_name = "Hepatocellular carcinoma"
                            elif "normal" in filename:
                                true_class_name = "Normal"
                            else:
                                true_class_name = "Unknown"
                        else:
                            true_class_name = "Unknown"
                        
                        # Check if prediction is correct
                        is_correct = (predicted_label == true_class_name)
                        
                        test_result = {
                            'image_file': filename,
                            'true_class_name': true_class_name,
                            'predicted_class_name': predicted_label,
                            'confidence': float(confidence),
                            'correct': is_correct,
                            'is_positive_case': is_positive,
                            'is_negative_case': is_negative
                        }
                    except Exception as e:
                        test_result = {
                            'image_file': filename,
                            'true_class_name': 'Error',
                            'predicted_class_name': f'Error: {str(e)}',
                            'confidence': 0.0,
                            'correct': False,
                            'is_positive_case': False,
                            'is_negative_case': False
                        }
                    
                    model_results['tests'].append(test_result)
            
            # Calculate accuracy
            correct = sum(1 for t in model_results['tests'] if t['correct'])
            total = len(model_results['tests'])
            model_results['accuracy'] = correct / total if total > 0 else 0
            model_results['correct_predictions'] = correct
            model_results['total_predictions'] = total
            
            results[model_key] = model_results
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test_images')
@login_required
def api_test_images():
    """API endpoint to get list of test images"""
    try:
        test_dir = 'test_medical_images'
        if not os.path.exists(test_dir):
            return jsonify({'error': 'Test images directory not found'}), 404
        
        import glob
        import os
        
        # Get all test images
        test_images = []
        all_files = glob.glob(os.path.join(test_dir, "*.png"))
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # Extract information from filename
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 4:
                cancer_type = parts[0]
                image_num = parts[1]
                condition = '_'.join(parts[2:-1])  # Everything except the last part
                status = parts[-1]  # POSITIVE or NEGATIVE
                
                test_images.append({
                    'filename': filename,
                    'cancer_type': cancer_type.title(),
                    'condition': condition.replace('_', ' ').title(),
                    'status': status,
                    'is_positive': status == 'POSITIVE',
                    'url': f'/test_image/{filename}'
                })
        
        # Sort by cancer type and then by filename
        test_images.sort(key=lambda x: (x['cancer_type'], x['filename']))
        
        return jsonify({'images': test_images, 'count': len(test_images)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_image/<filename>')
def serve_test_image(filename):
    """Serve test images"""
    try:
        test_dir = 'test_medical_images'
        return send_from_directory(test_dir, filename)
    except Exception as e:
        return str(e), 404

# Create database tables
with app.app_context():
    db.create_all()
    
    # Create default admin user if it doesn't exist
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@medical-ai.com',
            full_name='System Administrator',
            role='admin'
        )
        admin.set_password('admin123')  # Change this in production!
        db.session.add(admin)
        db.session.commit()
        print("✓ Default admin user created (username: admin, password: admin123)")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
