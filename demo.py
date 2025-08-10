"""
Flask App Demo Script
====================

This script demonstrates the Flask app functionality by sending a test image
to the prediction API and displaying the results.

Usage:
    python demo.py [image_path]
    
    If no image path is provided, it will try to find a sample from the test dataset.
"""

import os
import sys
import requests
import json
from pathlib import Path

def find_sample_image():
    """Find a sample image from the test dataset"""
    test_dir = Path("Dataset/test")
    
    if not test_dir.exists():
        return None
    
    # Look for any image in test folders
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            for image_file in class_dir.glob("*.jpg"):
                return str(image_file)
            for image_file in class_dir.glob("*.jpeg"):
                return str(image_file)
            for image_file in class_dir.glob("*.png"):
                return str(image_file)
    
    return None

def test_flask_app(image_path, app_url="http://localhost:5000"):
    """Test the Flask app with an image"""
    
    print(f"🐟 Testing Flask App at {app_url}")
    print("=" * 50)
    
    # Test 1: Health check
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{app_url}/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ App is healthy: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Device: {health_data['device']}")
            if health_data.get('model_type'):
                print(f"   Model type: {health_data['model_type']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app")
        print("   Make sure the app is running: python run_app.py")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Image prediction
    print(f"\n📸 Testing image prediction with: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{app_url}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'error' in result:
                print(f"❌ Prediction error: {result['error']}")
                return False
            
            print("✅ Prediction successful!")
            print(f"\n🏆 Top Prediction:")
            print(f"   Species: {result['predicted_class']}")
            print(f"   Confidence: {result['percentage']:.1f}%")
            
            print(f"\n📊 Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'][:3], 1):
                print(f"   {i}. {pred['class']}: {pred['percentage']:.1f}%")
            
            return True
            
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - prediction took too long")
        return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    print("🐟 Fish Classification Flask App Demo")
    print("=" * 40)
    
    # Get image path from command line or find sample
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("🔍 Looking for sample image...")
        image_path = find_sample_image()
        
        if not image_path:
            print("❌ No sample image found in Dataset/test/")
            print("\nUsage:")
            print("   python demo.py path/to/fish/image.jpg")
            print("\nOr make sure you have images in Dataset/test/ folders")
            sys.exit(1)
        
        print(f"✅ Found sample image: {image_path}")
    
    # Test the Flask app
    success = test_flask_app(image_path)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print(f"\n💡 You can also test manually:")
        print(f"   1. Open http://localhost:5000 in your browser")
        print(f"   2. Upload the image: {image_path}")
        print(f"   3. View the results")
    else:
        print("\n❌ Demo failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure Flask app is running: python run_app.py")
        print("   2. Check if model files exist: python test_model.py")
        print("   3. Verify the image file is valid")

if __name__ == '__main__':
    main()
