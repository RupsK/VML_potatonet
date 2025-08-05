#!/usr/bin/env python3
"""
Direct test of escalator analyzer with VLM
"""
import os
from escalator_vlm_analyzer import EscalatorVLMAnalyzer

def test_escalator_direct():
    print("🔍 Testing EscalatorVLMAnalyzer directly...")
    
    # Get token
    if os.path.exists("hf_token.txt"):
        with open("hf_token.txt", "r") as f:
            hf_token = f.read().strip()
        print(f"✅ Token loaded: {hf_token[:10]}...")
    else:
        print("❌ No token file")
        return False
    
    try:
        # Create analyzer
        print("🔄 Creating EscalatorVLMAnalyzer...")
        analyzer = EscalatorVLMAnalyzer(hf_token=hf_token)
        
        print(f"VLM Available: {analyzer.vlm_available}")
        print(f"VLM Processor: {analyzer.vlm_processor is not None}")
        print(f"VLM Model: {analyzer.vlm_model is not None}")
        
        if analyzer.vlm_available:
            print("✅ VLM is working properly!")
            
            # Test with a small video if available
            test_folder = "test_video"
            if os.path.exists(test_folder):
                import glob
                video_files = glob.glob(os.path.join(test_folder, "*.mp4"))
                if video_files:
                    test_video = video_files[0]
                    print(f"📹 Testing with: {test_video}")
                    
                    analyzer.max_frames_to_analyze = 3  # Small test
                    result = analyzer.analyze_escalator_vlm(test_video, "enhanced")
                    
                    if result and 'error' not in result:
                        print("✅ Analysis completed successfully!")
                        print(f"VLM Used: {result.get('vlm_used', False)}")
                        print(f"Crowding: {result['crowding_analysis']['crowding_detected']}")
                        print(f"Falling: {result['falling_analysis']['falling_detected']}")
                        print(f"Safety Alerts: {len(result['safety_alerts'])}")
                        return True
                    else:
                        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    print("⚠️ No test videos found")
                    return True
            else:
                print("⚠️ No test_video folder found")
                return True
        else:
            print("❌ VLM is not available")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_escalator_direct() 