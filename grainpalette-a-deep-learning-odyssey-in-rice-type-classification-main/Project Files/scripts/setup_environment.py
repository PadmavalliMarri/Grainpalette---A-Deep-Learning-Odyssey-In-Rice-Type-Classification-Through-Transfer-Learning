"""
Setup script for GrainPalette Rice Classification Project
This script helps set up the environment and download necessary components.
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "static/uploads",
        "models",
        "data",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_tensorflow():
    """Check TensorFlow installation"""
    print("🔍 Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check for GPU support
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU support detected!")
        else:
            print("💻 Running on CPU")
            
        return True
    except ImportError:
        print("❌ TensorFlow not found!")
        return False

def download_sample_data():
    """Provide instructions for downloading dataset"""
    print("\n📊 Dataset Setup Instructions:")
    print("=" * 50)
    print("1. Go to: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset")
    print("2. Download the dataset")
    print("3. Extract it to a folder named 'rice_dataset' in the project root")
    print("4. The structure should be:")
    print("   rice_dataset/")
    print("   ├── Arborio/")
    print("   ├── Basmati/")
    print("   ├── Ipsala/")
    print("   ├── Jasmine/")
    print("   └── Karacadag/")
    print("\nAlternatively, you can use your own rice images organized in the same structure.")

def main():
    """Main setup function"""
    print("🌾 GrainPalette Setup Script")
    print("=" * 30)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Check TensorFlow
    if not check_tensorflow():
        print("❌ Setup failed during TensorFlow check")
        return
    
    # Dataset instructions
    download_sample_data()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download the rice dataset (see instructions above)")
    print("2. Run 'python train_model.py' to train the model")
    print("3. Run 'python app.py' to start the web application")

if __name__ == "__main__":
    main()
