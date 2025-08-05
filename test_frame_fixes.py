#!/usr/bin/env python3
"""
Simple test to verify frame analysis fixes
"""

import os
from escalator_vlm_analyzer import EscalatorVLMAnalyzer

def test_frame_analysis_fixes():
    """Test the frame analysis fixes"""
    
    # Load token
    token = None
    if os.path.exists('hf_token.txt'):
        with open('hf_token.txt', 'r') as f:
            token = f.read().strip()
    
    print("Testing frame analysis fixes...")
    
    # Initialize analyzer
    analyzer = EscalatorVLMAnalyzer(token)
    print(f"VLM available: {analyzer.vlm_available}")
    
    # Test with a small video
    video_path = 'test_video/4.mp4'  # Smallest video
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    print(f"Testing with: {video_path}")
    
    # Run analysis with timeout
    try:
        result = analyzer.analyze_escalator_vlm(video_path)
        
        if 'error' in result:
            print(f"Analysis failed: {result['error']}")
        else:
            print("Analysis completed successfully!")
            print(f"Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"Frames analyzed: {result.get('total_frames_analyzed', 0)}")
            print(f"VLM used: {result.get('vlm_used', False)}")
            
            # Show summary
            if 'safety_summary' in result:
                print("\nSafety Summary:")
                print(result['safety_summary'][:500] + "..." if len(result['safety_summary']) > 500 else result['safety_summary'])
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frame_analysis_fixes() 