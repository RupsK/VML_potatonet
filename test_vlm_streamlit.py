#!/usr/bin/env python3
"""
Test VLM loading in Streamlit-like environment
"""
import os
import sys

def test_vlm_streamlit():
    print("🔍 Testing VLM in Streamlit-like environment...")
    
    # Get token
    if os.path.exists("hf_token.txt"):
        with open("hf_token.txt", "r") as f:
            hf_token = f.read().strip()
        print(f"✅ Token loaded: {hf_token[:10]}...")
    else:
        print("❌ No token file")
        return False
    
    try:
        # Import the analyzer
        from escalator_vlm_analyzer import EscalatorVLMAnalyzer
        
        # Create analyzer
        print("🔄 Creating EscalatorVLMAnalyzer...")
        analyzer = EscalatorVLMAnalyzer(hf_token=hf_token)
        
        print(f"VLM Available: {analyzer.vlm_available}")
        print(f"VLM Processor: {analyzer.vlm_processor is not None}")
        print(f"VLM Model: {analyzer.vlm_model is not None}")
        
        if analyzer.vlm_available:
            print("✅ VLM is working in Streamlit environment!")
            return True
        else:
            print("❌ VLM is not available")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_vlm_streamlit() 