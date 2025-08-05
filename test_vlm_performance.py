#!/usr/bin/env python3
"""
Performance test for VLM video analysis
Identifies bottlenecks and optimization opportunities
"""

import time
import os
from pathlib import Path
from escalator_vlm_analyzer import EscalatorVLMAnalyzer

def test_vlm_performance():
    """Test VLM performance and identify bottlenecks"""
    print("🔍 VLM Performance Analysis")
    print("=" * 50)
    
    # Load token
    token_path = "hf_token.txt"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            hf_token = f.read().strip()
        print(f"✅ Token loaded: {hf_token[:10]}...")
    else:
        hf_token = None
        print("❌ No token found")
    
    # Initialize analyzer with performance tracking
    print("\n🔄 Initializing VLM Analyzer...")
    init_start = time.time()
    analyzer = EscalatorVLMAnalyzer(hf_token)
    init_time = time.time() - init_start
    print(f"✅ Initialization time: {init_time:.2f}s")
    print(f"✅ VLM Available: {analyzer.vlm_available}")
    
    # Find smallest test video
    test_folder = "test_video"
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found")
        return
    
    video_files = list(Path(test_folder).glob("*.mp4"))
    if not video_files:
        print("❌ No video files found")
        return
    
    test_video = min(video_files, key=lambda x: x.stat().st_size)
    print(f"\n📹 Testing with: {test_video.name}")
    print(f"📊 File size: {test_video.stat().st_size / 1024:.1f} KB")
    
    # Test with different frame counts
    frame_counts = [3, 6, 12]
    
    for frame_count in frame_counts:
        print(f"\n🧪 Testing with {frame_count} frames...")
        analyzer.max_frames_to_analyze = frame_count
        
        start_time = time.time()
        result = analyzer.analyze_escalator_vlm(str(test_video), "enhanced")
        processing_time = time.time() - start_time
        
        if result and 'error' not in result:
            print(f"✅ {frame_count} frames processed in {processing_time:.2f}s")
            print(f"   - Average time per frame: {processing_time/frame_count:.2f}s")
            print(f"   - VLM used: {result['vlm_used']}")
            print(f"   - Total processing time: {result.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Detailed bottleneck analysis
    print(f"\n🔍 Detailed Performance Analysis:")
    print(f"   - VLM Model: {'microsoft/git-base' if analyzer.vlm_available else 'Not loaded'}")
    print(f"   - Model loaded: {analyzer.vlm_model is not None}")
    print(f"   - Processor loaded: {analyzer.vlm_processor is not None}")
    
    if analyzer.vlm_available:
        print(f"\n💡 Optimization Suggestions:")
        print(f"   - Reduce max_frames_to_analyze for faster processing")
        print(f"   - Use smaller VLM model for better speed")
        print(f"   - Implement frame skipping for longer videos")
        print(f"   - Add progress indicators for long processing")

if __name__ == "__main__":
    test_vlm_performance() 