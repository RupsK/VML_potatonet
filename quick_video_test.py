#!/usr/bin/env python3
"""
Quick Video Test Script
Tests video processing without Streamlit interface
"""

import os
import sys
from pathlib import Path

def test_video_processing():
    """Test video processing with a simple approach"""
    
    print("🎬 Quick Video Processing Test")
    print("=" * 50)
    
    # Check for test videos
    test_folder = "test_video"
    if not os.path.exists(test_folder):
        print(f"❌ Test video folder '{test_folder}' not found")
        return False
    
    # Find the smallest video file
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(Path(test_folder).glob(f"*{ext}"))
    
    if not video_files:
        print("❌ No video files found in test_video folder")
        return False
    
    # Use the smallest file
    test_video = min(video_files, key=lambda x: x.stat().st_size)
    print(f"📹 Testing with: {test_video.name} ({test_video.stat().st_size / 1024:.1f} KB)")
    
    try:
        # Test simple video processor
        print("\n🔄 Testing simple video processor...")
        from thermal_video_processor_simple import SimpleThermalVideoProcessor
        
        processor = SimpleThermalVideoProcessor()
        processor.max_frames_to_analyze = 3  # Very small for quick test
        
        result = processor.process_video(str(test_video), "Test analysis", "summary")
        
        if result and 'error' not in result:
            print("✅ Simple processor test PASSED")
            print(f"   - Processing time: {result['processing_time']:.2f}s")
            print(f"   - Frames analyzed: {result['total_frames_analyzed']}")
            print(f"   - Video duration: {result['video_info']['duration']:.2f}s")
            return True
        else:
            print(f"❌ Simple processor test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    
    print("\n🤖 Testing model loading...")
    
    try:
        # Test basic image processor
        from thermal_vlm_processor import ThermalImageProcessor
        processor = ThermalImageProcessor()
        print("✅ Basic processor loaded successfully")
        
        # Test simple video processor
        from thermal_video_processor_simple import SimpleThermalVideoProcessor
        video_processor = SimpleThermalVideoProcessor()
        print("✅ Simple video processor loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Starting Quick Video Test")
    print("=" * 50)
    
    # Test 1: Model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed. Check dependencies.")
        return
    
    # Test 2: Video processing
    if not test_video_processing():
        print("\n❌ Video processing test failed. Check video files.")
        return
    
    print("\n🎉 All tests PASSED!")
    print("💡 Your setup is ready for video analysis.")
    print("\n📋 Next steps:")
    print("   1. Run: streamlit run streamlit_app.py")
    print("   2. Select 'Video' mode in the sidebar")
    print("   3. Choose a test video and click 'Analyze'")

if __name__ == "__main__":
    main() 