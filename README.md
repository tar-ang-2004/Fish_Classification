# Fish Classification Project

An AI-powered fish species identification system using deep learning and computer vision with **PyTorch**. This internship project implements a Convolutional Neural Network (CNN) to classify 11 different types of fish species with high accuracy.

## 🐟 Features

- **Deep Learning Classification**: Uses PyTorch CNN and transfer learning (ResNet18) for accurate fish species identification
- **11 Fish Species**: Supports classification of multiple fish categories including sea bass, trout, shrimp, and more
- **Web Interface**: User-friendly Flask web application with drag-and-drop image upload
- **Real-time Predictions**: Instant classification results with confidence scores
- **Model Comparison**: Implements both custom CNN and transfer learning approaches
- **Comprehensive Analysis**: Detailed model evaluation with confusion matrix and classification reports

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook fish_classification.ipynb
```

Run all cells in the notebook to:
- Explore and preprocess the dataset
- Train both custom CNN and transfer learning models
- Evaluate model performance
- Save the best model for the Flask app

### 3. Run the Flask Web Application

#### Option A: Using the startup script (Recommended)
```bash
python run_app.py
```

#### Option B: Using the batch file (Windows)
```bash
start_app.bat
```

#### Option C: Direct Flask run
```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser.

### 4. Test the Model (Optional)

Before running the web app, you can test if everything is working:

```bash
python test_model.py
```

## 📁 Project Structure

```
P4/
├── Dataset/                   # Fish image dataset
│   ├── train/                 # Training images
│   ├── val/                   # Validation images
│   └── test/                  # Test images
├── templates/                 # HTML templates for Flask app
│   ├── base.html             # Base template
│   ├── index.html            # Main application page
│   ├── result.html           # Results page
│   ├── about.html            # About page
│   ├── 404.html              # Error pages
│   └── 500.html
├── static/                   # Static files (CSS, JS, uploads)
│   └── uploads/              # Temporary upload directory
├── fish_classification.ipynb  # Main Jupyter notebook
├── app.py                    # Flask web application
├── run_app.py               # Application startup script
├── test_model.py            # Model testing script
├── start_app.bat            # Windows batch file to start app
├── requirements.txt          # Python dependencies
├── model_info.json          # Model metadata (generated)
├── fish_classifier_model.pth # Trained PyTorch model (generated)
└── README.md                # This file
```

## 🐠 Supported Fish Species

1. Animal Fish
2. Animal Fish Bass
3. Fish Sea Food Black Sea Sprat
4. Fish Sea Food Gilt Head Bream
5. Fish Sea Food Horse Mackerel
6. Fish Sea Food Red Mullet
7. Fish Sea Food Red Sea Bream
8. Fish Sea Food Sea Bass
9. Fish Sea Food Shrimp
10. Fish Sea Food Striped Red Mullet
11. Fish Sea Food Trout

## 🧠 Model Architecture

### Custom CNN
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification

### Transfer Learning (ResNet18)
- Pre-trained ResNet18 backbone (frozen)
- Global Average Pooling
- Custom classification head
- Fine-tuned for fish classification

## 📊 Performance

The model achieves high accuracy on the test set with detailed evaluation metrics including:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix visualization
- Training/validation curves

## 🌐 Web Application Features

### Main Interface
- **Drag & Drop Upload**: Simply drag your fish image onto the upload area
- **Real-time Processing**: Instant image classification with progress indicators
- **Confidence Scores**: See how confident the AI is in its predictions
- **Top 3 Predictions**: View alternative classifications with probabilities
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### User Experience
- **Modern UI**: Dark theme with professional styling
- **Interactive Elements**: Smooth animations and hover effects
- **Error Handling**: Comprehensive error messages and validation
- **File Support**: JPG, PNG, GIF, BMP, WebP formats (up to 16MB)

### Technical Features
- **Model Loading**: Automatic model initialization on startup
- **Image Preprocessing**: Consistent image normalization and resizing
- **Performance Monitoring**: Health check endpoints for system status
- **Security**: File validation and secure upload handling

## 🛠️ Development

### Project Structure Details

```
app.py                 # Main Flask application
├── Model Loading      # PyTorch model initialization
├── Image Processing   # PIL/OpenCV preprocessing pipeline
├── Prediction API     # RESTful endpoints for classification
└── Error Handling     # Comprehensive exception management

templates/             # Jinja2 HTML templates
├── base.html         # Common layout and styling
├── index.html        # Main upload interface
├── result.html       # Prediction results display
├── about.html        # Project information
└── error pages       # 404, 500 error handling

static/               # Static assets
└── uploads/          # Temporary file storage
```

### API Endpoints

- `GET /` - Main application page
- `POST /predict` - AJAX prediction endpoint
- `POST /upload` - Form-based upload (fallback)
- `GET /about` - Project information
- `GET /api/health` - System health check

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=development     # Development/production mode
FLASK_DEBUG=1            # Enable debug mode
SECRET_KEY=your-key      # Flask secret key
MAX_CONTENT_LENGTH=16MB  # Maximum upload size
```

### Model Configuration
The application automatically loads:
- `fish_classifier_model.pth` - Trained PyTorch model
- `model_info.json` - Model metadata and class mappings

## 🎯 Usage Examples

### Basic Classification
1. Open `http://localhost:5000`
2. Drag an image to the upload area or click to browse
3. Wait for processing (usually < 2 seconds)
4. View prediction results with confidence scores

### API Usage
```python
import requests

# Upload image for classification
with open('fish_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    
result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['percentage']:.1f}%")
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: 2GB free space
- **GPU**: Optional but recommended (CUDA-compatible)

### Python Dependencies
```
torch>=2.0.1          # PyTorch deep learning framework
torchvision>=0.15.2   # Computer vision utilities
flask>=2.3.3          # Web framework
pillow>=10.0.0        # Image processing
numpy>=1.24.3         # Numerical computing
```
- Technology stack details

## 🔧 Technical Details

### Data Preprocessing
- Image resizing to 224x224 pixels
- Normalization (pixel values 0-1)
- Data augmentation for training:
  - Rotation, shifts, zoom
  - Horizontal flipping
  - Shear transformations

### Training Configuration
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss function: CrossEntropyLoss
- Callbacks: ReduceLROnPlateau, Early Stopping

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

## 💡 Usage Tips

1. **Image Quality**: Use clear, well-lit images of fish for best results
2. **File Formats**: Supports JPG, JPEG, PNG, and GIF formats
3. **Image Size**: Images are automatically resized to 224x224 pixels
4. **Model Performance**: Check the about page for current model accuracy

## 🛠️ Development

### Adding New Fish Species
1. Add new fish images to the dataset folders
2. Update the class labels in the notebook
3. Retrain the model
4. Update the Flask app if needed

### Model Improvements
- Experiment with different architectures (EfficientNet, Vision Transformer)
- Try different data augmentation techniques
- Implement ensemble methods
- Add more training data

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+
- Other dependencies listed in requirements.txt

## 🎯 Results Interpretation

The web application provides:
- **Top Prediction**: Most likely fish species with confidence percentage
- **All Predictions**: Ranked list of all possible species with probabilities
- **Confidence Score**: How certain the model is about its prediction

## 🚦 Running the Project

1. **First time setup**:
   ```bash
   pip install -r requirements.txt
   jupyter notebook fish_classification.ipynb
   # Run all cells to train the model
   python app.py
   ```

2. **Subsequent runs**:
   ```bash
   python app.py
   ```

## 📝 Notes

- The model files (`fish_classifier_model.pth` and `model_info.json`) are generated after running the Jupyter notebook
- Training time depends on your hardware (GPU recommended)
- The Flask app will show a warning if the model is not found

## 🎓 Educational Value

This project demonstrates:
- End-to-end machine learning pipeline
- Computer vision and image classification
- Web application development with Flask
- Model evaluation and comparison
- Transfer learning techniques
- Data visualization and analysis

Perfect for learning about practical AI applications in marine biology and species identification!
