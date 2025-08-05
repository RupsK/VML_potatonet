# test_simple_video.py
"""
Test script for the simplified video processor
"""

import os
from thermal_video_processor_simple import SimpleThermalVideoProcessor

def test_simple_processor():
    """Test the simplified video processor"""
    print("🧪 Testing Simplified Video Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = SimpleThermalVideoProcessor()
    
    # Test with existing video files
    test_videos = []
    
    # Check for temporary files
    temp_files = [f for f in os.listdir('.') if f.startswith('temp_') and f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    test_videos.extend(temp_files)
    
    # Check test video folder
    if os.path.exists("test_video"):
        test_videos.extend([f"test_video/{f}" for f in os.listdir("test_video") if f.endswith(('.mp4', '.avi'))])
    
    if not test_videos:
        print("❌ No test videos found")
        return
    
    print(f"📁 Found {len(test_videos)} test videos")
    
    for video_path in test_videos:
        print(f"\n--- Testing {video_path} ---")
        
        try:
            # Process video
            result = processor.process_video(
                video_path,
                custom_prompt="Analyze this thermal video for temperature patterns and anomalies.",
                analysis_mode="summary"
            )
            
            if result and 'error' not in result:
                print(f"✅ Success!")
                print(f"   - Duration: {result['video_info']['duration']:.2f}s")
                print(f"   - Frames analyzed: {result['total_frames_analyzed']}")
                print(f"   - Processing time: {result['processing_time']:.2f}s")
                print(f"   - Summary: {result['video_summary'][:100]}...")
                
                # Show frame analysis
                if result['frame_analyses']:
                    print(f"   - Frame analysis examples:")
                    for i, analysis in enumerate(result['frame_analyses'][:3]):
                        print(f"     Frame {analysis['frame_number']}: {analysis['caption']}")
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                print(f"❌ Failed: {error_msg}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_simple_processor() 