# test_app_simple.py
"""
Simple test to verify the app can run without optional dependencies
"""

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit imported")
    except Exception as e:
        print(f"❌ streamlit failed: {e}")
        return False
    
    try:
        import torch
        print("✅ torch imported")
    except Exception as e:
        print(f"❌ torch failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported")
    except Exception as e:
        print(f"❌ PIL failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported")
    except Exception as e:
        print(f"❌ numpy failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported")
    except Exception as e:
        print(f"❌ matplotlib failed: {e}")
        return False
    
    # Test seaborn (optional)
    try:
        import seaborn as sns
        print("✅ seaborn imported")
    except ImportError:
        print("⚠️ seaborn not available (optional)")
    except Exception as e:
        print(f"❌ seaborn failed: {e}")
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except Exception as e:
        print(f"❌ pandas failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv imported")
    except Exception as e:
        print(f"❌ opencv failed: {e}")
        return False
    
    # Test our custom modules
    try:
        from thermal_vlm_processor import ThermalImageProcessor
        print("✅ thermal_vlm_processor imported")
    except Exception as e:
        print(f"❌ thermal_vlm_processor failed: {e}")
        return False
    
    try:
        from thermal_video_processor_simple import SimpleThermalVideoProcessor
        print("✅ thermal_video_processor_simple imported")
    except Exception as e:
        print(f"❌ thermal_video_processor_simple failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without full app"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test video processor
        from thermal_video_processor_simple import SimpleThermalVideoProcessor
        processor = SimpleThermalVideoProcessor()
        print("✅ Video processor created")
        
        # Test supported formats
        formats = processor.get_supported_formats()
        print(f"✅ Supported formats: {formats}")
        
        # Test analysis modes
        modes = processor.get_analysis_modes()
        print(f"✅ Analysis modes: {modes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False

def main():
    print("🔧 Testing Thermal Video Analysis Setup")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("🎉 All tests passed!")
        print("\n🚀 You can now run:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed.")
        print("Please check the error messages above.")
    
    print("\n📋 Next steps:")
    print("1. Run: streamlit run streamlit_app.py")
    print("2. Select 'Video' in the sidebar")
    print("3. Upload your thermal video file")
    print("4. Choose analysis settings and click analyze")

if __name__ == "__main__":
    main() 