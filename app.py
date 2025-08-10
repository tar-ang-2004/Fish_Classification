"""
Fish Classification Flask Web Application
=========================================

A modern web application for classifying fish species using deep learning.
Supports 11 different fish species with high accuracy using PyTorch CNN models.

Features:
- Drag & drop image upload
- Real-time prediction with confidence scores
- Support for both custom CNN and transfer learning models
- Modern responsive UI with dark theme
- Error handling and validation

Author: Your Name
Date: 2025
"""

import os
import json
import io
import base64
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_CONFIG', 'development')
if config_name == 'development':
    from config import DevelopmentConfig
    app.config.from_object(DevelopmentConfig)
elif config_name == 'production':
    from config import ProductionConfig
    app.config.from_object(ProductionConfig)
else:
    from config import DevelopmentConfig
    app.config.from_object(DevelopmentConfig)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'})

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model architectures (copied from notebook)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions
        # After 4 pooling operations: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Global variables for model and metadata
model = None
model_info = None
transform = None

def load_model():
    """Load the trained model and metadata"""
    global model, model_info, transform
    
    try:
        # Load model info
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"Loaded model info: {model_info['model_type']}")
        
        # Load model checkpoint
        checkpoint = torch.load('fish_classifier_model.pth', map_location=device)
        
        # Initialize model based on type
        if checkpoint['model_type'] == 'Transfer Learning':
            model = TransferLearningModel(checkpoint['num_classes'])
        else:
            model = CustomCNN(checkpoint['num_classes'])
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Define image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ Model loaded successfully: {checkpoint['model_type']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(device)
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_image(image):
    """Make prediction on preprocessed image"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            
            # Get predicted class
            predicted_idx = predicted.item()
            predicted_class = model_info['idx_to_class'][str(predicted_idx)]
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities[0], 3)
            top_predictions = []
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = model_info['idx_to_class'][str(idx.item())]
                top_predictions.append({
                    'rank': i + 1,
                    'class': class_name,
                    'confidence': prob.item(),
                    'percentage': prob.item() * 100
                })
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'percentage': confidence_score * 100,
                'top_predictions': top_predictions,
                'all_probabilities': all_probs.tolist()
            }
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

@app.route('/')
def index():
    """Main page"""
    if model is None:
        flash('Model not loaded. Please check if the model file exists.', 'error')
    
    return render_template('index.html', 
                         model_info=model_info,
                         classes=model_info['class_labels'] if model_info else [])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_image(image)
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        result['image_data'] = f"data:image/jpeg;base64,{img_str}"
        result['status'] = 'success'
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload via form"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image.', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and process image
        image = Image.open(filepath)
        
        # Make prediction
        result = predict_image(image)
        
        # Convert image to base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        result['image_data'] = f"data:image/jpeg;base64,{img_str}"
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', result=result, model_info=model_info)
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', model_info=model_info)

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'model_type': model_info['model_type'] if model_info else None
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.warning("Failed to load model. Application will start but predictions will not work.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
