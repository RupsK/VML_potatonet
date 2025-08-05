#!/usr/bin/env python3
"""
Test script for improved VLM insights processing
Demonstrates the enhanced error correction and safety-focused analysis
"""

from escalator_vlm_analyzer import EscalatorVLMAnalyzer
import time

def test_vlm_description_improvements():
    """Test the improved VLM description processing"""
    print("Testing Improved VLM Description Processing")
    print("=" * 50)
    
    # Create analyzer instance
    analyzer = EscalatorVLMAnalyzer()
    
    # Test cases with problematic VLM outputs
    test_cases = [
        "The escalator is seen in this undated image with people waiting",
        "The woman was hit by a woman who was hit by an electric vehicle",
        "The scary woman was knocked down the escalator",
        "This is an outdoor scene with escalator",
        "Passengers wait on the escalator at the metro station",
        "A busy escalator with many people and bags",
        "The image shows a crowded escalator with potential safety concerns"
    ]
    
    print("Original VLM Outputs vs Improved Outputs:")
    print("-" * 50)
    
    for i, original in enumerate(test_cases, 1):
        improved = analyzer._simplify_vlm_description(original)
        print(f"Test {i}:")
        print(f"  Original: {original}")
        print(f"  Improved: {improved}")
        print()
    
    print("Safety Assessment:")
    print("-" * 30)
    
    # Test safety keyword detection
    safety_keywords = ['fall', 'falling', 'crowd', 'crowded', 'safety', 'risk', 'concern']
    
    for i, original in enumerate(test_cases, 1):
        improved = analyzer._simplify_vlm_description(original)
        has_safety_concerns = any(keyword in improved.lower() for keyword in safety_keywords)
        safety_status = "⚠️ SAFETY CONCERN" if has_safety_concerns else "✅ NORMAL"
        print(f"Frame {i}: {safety_status} - {improved}")

def test_video_analysis_with_improvements():
    """Test the complete video analysis with improved VLM processing"""
    print("\n" + "=" * 60)
    print("Testing Complete Video Analysis with Improved VLM")
    print("=" * 60)
    
    # Find test video
    import os
    from pathlib import Path
    
    test_folder = "test_video"
    if not os.path.exists(test_folder):
        print(f"Test folder '{test_folder}' not found")
        return False
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(Path(test_folder).glob(f"*{ext}"))
    
    if not video_files:
        print("No video files found for testing")
        return False
    
    # Use the smallest file
    test_video = min(video_files, key=lambda x: x.stat().st_size)
    print(f"Testing with: {test_video.name}")
    
    try:
        analyzer = EscalatorVLMAnalyzer()
        analyzer.max_frames_to_analyze = 3  # Small test for faster processing
        
        print("Starting analysis...")
        start_time = time.time()
        
        result = analyzer.analyze_escalator_vlm(str(test_video), "enhanced")
        
        processing_time = time.time() - start_time
        
        if result and 'error' not in result:
            print(f"\n✅ Analysis completed successfully in {processing_time:.2f}s")
            print(f"Frames analyzed: {result['total_frames_analyzed']}")
            print(f"VLM used: {result['vlm_used']}")
            
            # Display the improved AI insights
            if 'safety_summary' in result:
                print("\n" + "=" * 40)
                print("IMPROVED AI MODEL INSIGHTS")
                print("=" * 40)
                print(result['safety_summary'])
            
            # Display safety alerts
            if 'safety_alerts' in result:
                print("\n" + "=" * 30)
                print("SAFETY ALERTS")
                print("=" * 30)
                for alert in result['safety_alerts']:
                    print(f"• {alert}")
        else:
            print(f"❌ Analysis failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Improved VLM Insights Processing")
    print("This script demonstrates the enhanced error correction and safety-focused analysis")
    print()
    
    # Test the description improvements
    test_vlm_description_improvements()
    
    # Test complete video analysis
    test_video_analysis_with_improvements()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("The improvements include:")
    print("• Better error correction for VLM outputs")
    print("• Enhanced safety-focused prompts")
    print("• Improved description formatting")
    print("• Better fallback handling for failed analyses")
    print("• Safety keyword detection and assessment") 