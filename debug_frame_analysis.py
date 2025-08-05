#!/usr/bin/env python3
"""
Debug script to identify frame analysis issues in escalator VLM analyzer
"""

import cv2
import numpy as np
import os
from escalator_vlm_analyzer import EscalatorVLMAnalyzer

def debug_frame_analysis():
    """Debug the frame analysis step by step"""
    
    # Load token
    token = None
    if os.path.exists('hf_token.txt'):
        with open('hf_token.txt', 'r') as f:
            token = f.read().strip()
    
    print("1. Initializing analyzer...")
    analyzer = EscalatorVLMAnalyzer(token)
    print(f"VLM available: {analyzer.vlm_available}")
    
    # Test video path
    video_path = 'test_video/1.mp4'
    print(f"\n2. Testing video: {video_path}")
    
    # Validate video
    print("3. Validating video...")
    is_valid = analyzer._validate_video(video_path)
    print(f"Video valid: {is_valid}")
    
    if not is_valid:
        print("Video validation failed!")
        return
    
    # Extract metadata
    print("4. Extracting video metadata...")
    try:
        metadata = analyzer._extract_video_metadata(video_path)
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Metadata extraction error: {e}")
        return
    
    # Extract frames
    print("5. Extracting frames...")
    try:
        frames = analyzer._extract_frames(video_path)
        print(f"Extracted {len(frames)} frames")
        if frames:
            print(f"First frame shape: {frames[0][1].shape}")
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return
    
    # Test basic frame analysis
    print("6. Testing basic frame analysis...")
    if frames:
        try:
            frame_number, frame, timestamp = frames[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Test individual components
            print("Testing crowding score calculation...")
            crowding_score = analyzer._calculate_crowding_score(gray)
            print(f"Crowding score: {crowding_score}")
            
            print("Testing falling detection...")
            falling_detection = analyzer._detect_falling_basic(gray, frame)
            print(f"Falling detection: {falling_detection}")
            
            print("Testing basic description...")
            basic_desc = analyzer._generate_basic_description(frame, gray)
            print(f"Basic description: {basic_desc}")
            
        except Exception as e:
            print(f"Basic frame analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test VLM frame analysis
    print("7. Testing VLM frame analysis...")
    if frames and analyzer.vlm_available:
        try:
            frame_number, frame, timestamp = frames[0]
            vlm_analysis = analyzer._analyze_frame_vlm(frame)
            print(f"VLM analysis keys: {list(vlm_analysis.keys())}")
            print(f"VLM description: {vlm_analysis.get('vlm_description', 'N/A')}")
            print(f"VLM crowding score: {vlm_analysis.get('crowding_vlm_score', 'N/A')}")
            print(f"VLM falling score: {vlm_analysis.get('falling_vlm_score', 'N/A')}")
        except Exception as e:
            print(f"VLM frame analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test full frame analysis
    print("8. Testing full frame analysis...")
    try:
        frame_analyses = analyzer._analyze_frames_vlm(frames[:3])  # Test with first 3 frames
        print(f"Analyzed {len(frame_analyses)} frames")
        if frame_analyses:
            print(f"First analysis keys: {list(frame_analyses[0].keys())}")
    except Exception as e:
        print(f"Full frame analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_frame_analysis() 