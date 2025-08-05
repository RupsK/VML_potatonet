# test_video_processing.py
"""
Test script for thermal video processing capabilities
This script demonstrates how to use the ThermalVideoProcessor
"""

import cv2
import numpy as np
import os
from pathlib import Path
from thermal_video_processor import ThermalVideoProcessor

def create_test_video(output_path="test_video/sample_thermal.mp4", duration=5, fps=10):
    """
    Create a sample thermal video for testing
    """
    print(f"Creating test video: {output_path}")
    
    # Video parameters
    width, height = 640, 480
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frames with simulated thermal patterns
    for frame_num in range(duration * fps):
        # Create base thermal image
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simulate thermal patterns
        # Hot region (red) moving across the frame
        hot_x = int((frame_num / (duration * fps)) * width)
        cv2.circle(frame, (hot_x, height//2), 50, (0, 0, 255), -1)  # Red circle (hot)
        
        # Cold region (blue) in the background
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle (cold)
        
        # Add some noise to simulate thermal sensor noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add text overlay
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_num/fps:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")
    return output_path

def test_video_processor():
    """
    Test the video processor with a sample video
    """
    print("Testing Thermal Video Processor...")
    
    # Create test video if it doesn't exist
    test_video_path = "test_video/sample_thermal.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path)
    
    # Initialize processor
    processor = ThermalVideoProcessor()
    
    # Test different analysis modes
    analysis_modes = ["summary", "key_frames", "comprehensive"]
    
    for mode in analysis_modes:
        print(f"\n--- Testing {mode} mode ---")
        
        try:
            # Process video
            result = processor.process_video(
                test_video_path, 
                custom_prompt="Analyze this thermal video. Describe the temperature patterns, motion, and any anomalies.",
                analysis_mode=mode
            )
            
            if result and 'error' not in result:
                print(f"✅ {mode} mode successful!")
                print(f"   - Frames analyzed: {result['total_frames_analyzed']}")
                print(f"   - Processing time: {result['processing_time']:.2f}s")
                print(f"   - Video duration: {result['video_info']['duration']:.2f}s")
                
                # Show video summary
                print(f"   - Summary: {result['video_summary'][:200]}...")
                
                # Show temporal analysis if available
                if 'temporal_analysis' in result and 'error' not in result['temporal_analysis']:
                    temp_trend = result['temporal_analysis'].get('temperature_trend', {})
                    if temp_trend:
                        print(f"   - Temperature trend: {temp_trend.get('trend', 'unknown')}")
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                print(f"❌ {mode} mode failed: {error_msg}")
                
        except Exception as e:
            print(f"❌ {mode} mode failed with exception: {e}")
    
    print("\n--- Test completed ---")

def test_supported_formats():
    """
    Test supported video formats
    """
    print("\nTesting supported video formats...")
    
    processor = ThermalVideoProcessor()
    supported_formats = processor.get_supported_formats()
    
    print("Supported video formats:")
    for fmt in supported_formats:
        print(f"  - {fmt}")
    
    print(f"\nTotal supported formats: {len(supported_formats)}")

def test_analysis_modes():
    """
    Test available analysis modes
    """
    print("\nTesting analysis modes...")
    
    processor = ThermalVideoProcessor()
    analysis_modes = processor.get_analysis_modes()
    
    print("Available analysis modes:")
    for mode in analysis_modes:
        print(f"  - {mode}")
    
    print(f"\nTotal analysis modes: {len(analysis_modes)}")

if __name__ == "__main__":
    print("🔥 Thermal Video Processing Test Suite")
    print("=" * 50)
    
    # Test supported formats and modes
    test_supported_formats()
    test_analysis_modes()
    
    # Test video processor
    test_video_processor()
    
    print("\n🎉 All tests completed!")
    print("\nTo use the video processing in the Streamlit app:")
    print("1. Run: streamlit run streamlit_app.py")
    print("2. Select 'Video' as input type")
    print("3. Upload a video file or use the test video")
    print("4. Choose analysis mode and settings")
    print("5. Click analyze to process the video") 