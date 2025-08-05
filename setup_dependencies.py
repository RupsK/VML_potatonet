# setup_dependencies.py
"""
Setup script to check and install required dependencies
"""

import subprocess
import sys
import importlib

def check_and_install_package(package_name, install_name=None):
    """Check if a package is installed, install if not"""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def main():
    print("🔧 Setting up Thermal Video Analysis Dependencies")
    print("=" * 60)
    
    # Core dependencies
    core_packages = [
        ("streamlit", "streamlit"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("protobuf", "protobuf"),
    ]
    
    print("\n📦 Checking core dependencies...")
    all_installed = True
    
    for package, install_name in core_packages:
        if not check_and_install_package(package, install_name):
            all_installed = False
    
    # Optional dependencies
    print("\n📦 Checking optional dependencies...")
    optional_packages = [
        ("psutil", "psutil"),  # For system resource monitoring
        ("dotenv", "python-dotenv"),  # For environment variables
    ]
    
    for package, install_name in optional_packages:
        check_and_install_package(package, install_name)
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        all_installed = False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
        all_installed = False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        all_installed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_installed:
        print("🎉 All core dependencies are installed!")
        print("\n🚀 You can now run:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️ Some dependencies failed to install.")
        print("Please check the error messages above and try installing manually.")
    
    print("\n📋 Next steps:")
    print("1. Run: streamlit run streamlit_app.py")
    print("2. Select 'Video' in the sidebar")
    print("3. Upload your thermal video file")
    print("4. Choose analysis settings and click analyze")

if __name__ == "__main__":
    main() 