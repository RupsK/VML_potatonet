# debug_video_upload.py
"""
Debug script to diagnose video upload and processing issues
"""

import cv2
import os
import sys
from pathlib import Path

def check_video_file(video_path):
    """Check if video file is valid and get its properties"""
    print(f"🔍 Checking video file: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ File does not exist: {video_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(video_path)
    print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
    
    if file_size == 0:
        print("❌ File is empty")
        return False
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ OpenCV cannot open the video file")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"✅ Video properties:")
        print(f"   - FPS: {fps}")
        print(f"   - Frame count: {frame_count}")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - Duration: {duration:.2f} seconds")
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read first frame")
            cap.release()
            return False
        
        print(f"✅ First frame read successfully: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error reading video: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    print("\n💻 System Resources:")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   - Available RAM: {memory.available / (1024**3):.2f} GB")
        print(f"   - Total RAM: {memory.total / (1024**3):.2f} GB")
        print(f"   - Memory usage: {memory.percent}%")
    except ImportError:
        print("   - psutil not available, cannot check memory")
    
    # Check disk space
    try:
        disk = psutil.disk_usage('.')
        print(f"   - Available disk space: {disk.free / (1024**3):.2f} GB")
    except:
        print("   - Cannot check disk space")

def test_simple_video_processing(video_path):
    """Test basic video processing without VLM"""
    print(f"\n🧪 Testing basic video processing...")
    
    try:
        cap = cv2.VideoCapture(video_path)
        frames_processed = 0
        max_frames = 10  # Only process first 10 frames for testing
        
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simple processing - just convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_processed += 1
            
            if frames_processed % 5 == 0:
                print(f"   - Processed frame {frames_processed}")
        
        cap.release()
        print(f"✅ Successfully processed {frames_processed} frames")
        return True
        
    except Exception as e:
        print(f"❌ Error in basic processing: {e}")
        return False

def create_minimal_test_video():
    """Create a very simple test video"""
    print("\n🎬 Creating minimal test video...")
    
    output_path = "test_video/minimal_test.mp4"
    os.makedirs("test_video", exist_ok=True)
    
    # Create a simple 2-second video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (320, 240))
    
    for i in range(10):  # 10 frames at 5 fps = 2 seconds
        # Create a simple frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Add a moving rectangle
        x = int((i / 10) * 280)
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (0, 0, 255), -1)
        out.write(frame)
    
    out.release()
    print(f"✅ Created minimal test video: {output_path}")
    return output_path

def main():
    print("🔧 Video Upload Debug Tool")
    print("=" * 50)
    
    # Check system resources
    check_system_resources()
    
    # Check for uploaded video files
    temp_files = [f for f in os.listdir('.') if f.startswith('temp_') and f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if temp_files:
        print(f"\n📁 Found temporary video files: {temp_files}")
        for temp_file in temp_files:
            print(f"\n--- Checking {temp_file} ---")
            if check_video_file(temp_file):
                test_simple_video_processing(temp_file)
    else:
        print("\n📁 No temporary video files found")
    
    # Check test video folder
    test_video_folder = "test_video"
    if os.path.exists(test_video_folder):
        test_videos = list(Path(test_video_folder).glob("*.mp4")) + list(Path(test_video_folder).glob("*.avi"))
        if test_videos:
            print(f"\n📁 Found test videos: {[v.name for v in test_videos]}")
            for video in test_videos:
                print(f"\n--- Checking {video.name} ---")
                if check_video_file(str(video)):
                    test_simple_video_processing(str(video))
        else:
            print("\n📁 No test videos found")
    
    # Create minimal test video if needed
    print("\n" + "=" * 50)
    print("💡 Recommendations:")
    print("1. If video processing is slow, try reducing video resolution")
    print("2. If memory issues occur, try shorter videos")
    print("3. If format issues occur, convert to MP4")
    print("4. Check that you have sufficient disk space")
    print("5. Try the 'summary' analysis mode instead of 'comprehensive'")

if __name__ == "__main__":
    import numpy as np
    main() 