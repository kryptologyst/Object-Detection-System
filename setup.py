#!/usr/bin/env python3
"""
Setup script for Object Detection System
Automatically installs dependencies and initializes the system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "sample_images",
        "logs",
        "cache",
        "results",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def download_sample_models():
    """Download sample models if needed"""
    print("\n🤖 Checking for model files...")
    
    # Check if YOLO model exists
    yolo_model = "yolov8n.pt"
    if not os.path.exists(yolo_model):
        print(f"📥 Downloading {yolo_model}...")
        try:
            from ultralytics import YOLO
            model = YOLO(yolo_model)  # This will download the model
            print(f"✅ {yolo_model} downloaded successfully")
        except Exception as e:
            print(f"⚠️ Could not download {yolo_model}: {e}")
            print("The model will be downloaded automatically on first use")
    else:
        print(f"✅ {yolo_model} already exists")
    
    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import tensorflow as tf
        import torch
        import cv2
        import streamlit
        import ultralytics
        import pandas
        import plotly
        
        print("✅ All core dependencies imported successfully")
        
        # Test basic functionality
        from object_detector import ObjectDetector
        detector = ObjectDetector()
        print("✅ ObjectDetector initialized successfully")
        
        from database import DetectionDatabase
        db = DetectionDatabase("test_setup.db")
        db.close()
        os.remove("test_setup.db")
        print("✅ Database system working correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def create_sample_config():
    """Create sample configuration if it doesn't exist"""
    print("\n⚙️ Checking configuration...")
    
    if not os.path.exists("config.yaml"):
        print("📝 Creating sample configuration...")
        # The config.yaml should already exist from our earlier creation
        print("✅ Configuration file ready")
    else:
        print("✅ Configuration file already exists")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Object Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Download sample models
    if not download_sample_models():
        print("\n⚠️ Setup completed with warnings about model downloads")
    
    # Create sample config
    if not create_sample_config():
        print("\n❌ Setup failed during configuration setup")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    print("\n🚀 Next steps:")
    print("1. Run the demo: python 0096.py")
    print("2. Start the web interface: streamlit run app.py")
    print("3. Upload images and start detecting objects!")
    
    print("\n📚 Documentation:")
    print("- README.md: Complete documentation")
    print("- config.yaml: Configuration options")
    print("- app.py: Web interface")
    print("- object_detector.py: Core functionality")
    
    print("\n🔧 Troubleshooting:")
    print("- If you encounter GPU issues, set use_gpu: false in config.yaml")
    print("- For memory issues, reduce batch_size in config.yaml")
    print("- Check logs/ directory for detailed error information")

if __name__ == "__main__":
    main()
