#!/usr/bin/env python3
"""
Diagnostic Script for Thermal Image Processing Issues
"""

import sys
import os
import time
import traceback

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    modules_to_test = [
        "streamlit",
        "torch", 
        "cv2",
        "numpy",
        "PIL",
        "matplotlib",
        "pandas"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return failed_imports

def check_custom_modules():
    """Check if custom modules can be imported"""
    print("\n🔍 Checking custom modules...")
    
    custom_modules = [
        "thermal_vlm_processor",
        "thermal_video_processor_simple",
        "thermal_smolvlm_processor"
    ]
    
    failed_modules = []
    
    for module in custom_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_modules.append(module)
    
    return failed_modules

def check_video_files():
    """Check if video files exist and are accessible"""
    print("\n🔍 Checking video files...")
    
    test_folder = "test_video"
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found")
        return False
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        files = list(Path(test_folder).glob(f"*{ext}"))
        video_files.extend(files)
    
    if not video_files:
        print("❌ No video files found")
        return False
    
    print(f"✅ Found {len(video_files)} video files:")
    for video in video_files:
        size_kb = video.stat().st_size / 1024
        print(f"   - {video.name} ({size_kb:.1f} KB)")
    
    return True

def test_simple_processor():
    """Test the simple video processor without heavy models"""
    print("\n🔍 Testing simple video processor...")
    
    try:
        from thermal_video_processor_simple import SimpleThermalVideoProcessor
        
        processor = SimpleThermalVideoProcessor()
        print("✅ Simple processor created successfully")
        
        # Test with smallest video
        test_folder = "test_video"
        video_files = []
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(Path(test_folder).glob(f"*{ext}"))
        
        if video_files:
            test_video = min(video_files, key=lambda x: x.stat().st_size)
            print(f"📹 Testing with: {test_video.name}")
            
            start_time = time.time()
            result = processor.process_video(str(test_video), "Test", "summary")
            processing_time = time.time() - start_time
            
            if result and 'error' not in result:
                print(f"✅ Video processing successful ({processing_time:.2f}s)")
                return True
            else:
                print(f"❌ Video processing failed: {result.get('error', 'Unknown')}")
                return False
        
    except Exception as e:
        print(f"❌ Simple processor test failed: {e}")
        traceback.print_exc()
        return False

def check_system_resources():
    """Check system resources"""
    print("\n🔍 Checking system resources...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ RAM: {memory.available / (1024**3):.1f} GB available")
        
        disk = psutil.disk_usage('.')
        print(f"✅ Disk: {disk.free / (1024**3):.1f} GB free")
        
    except ImportError:
        print("⚠️ psutil not available - can't check system resources")

def main():
    """Main diagnostic function"""
    print("🔧 Thermal Image Processing Diagnostic")
    print("=" * 50)
    
    # Check imports
    failed_imports = check_imports()
    
    # Check custom modules
    failed_modules = check_custom_modules()
    
    # Check video files
    videos_ok = check_video_files()
    
    # Check system resources
    check_system_resources()
    
    # Test simple processor
    processor_ok = test_simple_processor()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
    else:
        print("✅ All imports successful")
    
    if failed_modules:
        print(f"❌ Failed custom modules: {', '.join(failed_modules)}")
    else:
        print("✅ All custom modules successful")
    
    if videos_ok:
        print("✅ Video files accessible")
    else:
        print("❌ Video files not accessible")
    
    if processor_ok:
        print("✅ Simple processor working")
    else:
        print("❌ Simple processor failed")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if failed_imports:
        print("   1. Install missing packages: pip install " + " ".join(failed_imports))
    
    if failed_modules:
        print("   2. Check that all .py files are in the current directory")
    
    if not videos_ok:
        print("   3. Ensure test_video folder exists with video files")
    
    if not processor_ok:
        print("   4. Try restarting the application")
        print("   5. Check for error messages in the console")
    
    if not failed_imports and not failed_modules and videos_ok and processor_ok:
        print("   🎉 Everything looks good! Try the Streamlit app again.")

if __name__ == "__main__":
    from pathlib import Path
    main() 