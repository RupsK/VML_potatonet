#!/usr/bin/env python3
"""
Test script for the updated VLM-enhanced escalator analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from escalator_vlm_analyzer import EscalatorVLMAnalyzer
import cv2
import numpy as np

def test_vlm_descriptions():
    """Test the updated VLM description generation"""
    
    # Get token
    hf_token = None
    if os.path.exists("hf_token.txt"):
        with open("hf_token.txt", "r") as f:
            hf_token = f.read().strip()
        print(f"✅ Token loaded: {hf_token[:10]}...")
    else:
        print("❌ No token file found")
        return
    
    # Initialize analyzer
    print("🔧 Initializing VLM-enhanced escalator analyzer...")
    analyzer = EscalatorVLMAnalyzer(hf_token=hf_token)
    
    if not analyzer.vlm_available:
        print("❌ VLM not available - cannot test descriptions")
        return
    
    print("✅ VLM initialized successfully!")
    
    # Test video path
    test_video = "test_video/test1.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    print(f"📹 Testing with video: {test_video}")
    
    # Analyze video
    print("🤖 Running VLM analysis...")
    result = analyzer.analyze_escalator_vlm(test_video, "enhanced")
    
    if result and 'error' not in result:
        print("\n📋 VLM Analysis Results:")
        print("=" * 50)
        
        # Show frame analyses
        for i, analysis in enumerate(result['frame_analyses'][:5]):  # Show first 5 frames
            print(f"\n🎞️ Frame {analysis['frame_number']} (t={analysis['timestamp']:.2f}s):")
            print(f"   🤖 AI Description: {analysis.get('vlm_description', 'No description')}")
            print(f"   👥 Crowding Score: {analysis['crowding_score']:.1f}/100")
            print(f"   🤖 VLM Crowding: {analysis.get('crowding_vlm_score', 0):.1f}/100")
            print(f"   📦 Falling Risk: {analysis['falling_detection']['falling_probability']:.1f}/100")
            print(f"   🤖 VLM Falling: {analysis.get('falling_vlm_score', 0):.1f}/100")
        
        # Show safety alerts
        print(f"\n🚨 Safety Alerts:")
        for alert in result['safety_alerts']:
            print(f"   • {alert}")
        
        # Show summary
        print(f"\n📊 Summary:")
        print(f"   • VLM Used: {result.get('vlm_used', False)}")
        print(f"   • Processing Time: {result['processing_time']:.2f}s")
        print(f"   • Frames Analyzed: {result['total_frames_analyzed']}")
        
    else:
        print("❌ Analysis failed")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_vlm_descriptions() 