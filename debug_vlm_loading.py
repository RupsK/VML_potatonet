#!/usr/bin/env python3
"""
Debug script to test VLM loading step by step
"""

import os
import sys

def test_vlm_loading():
    print("=== VLM Loading Debug Test ===")
    
    # Step 1: Check if token file exists
    print("\n1. Checking token file...")
    if os.path.exists("hf_token.txt"):
        try:
            with open("hf_token.txt", "r") as f:
                token = f.read().strip()
            print(f"✅ Token file found: {token[:10]}...")
        except Exception as e:
            print(f"❌ Error reading token file: {e}")
            return
    else:
        print("❌ Token file not found")
        return
    
    # Step 2: Check imports
    print("\n2. Checking imports...")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        print("✅ Transformers and torch imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Step 3: Test processor loading
    print("\n3. Testing processor loading...")
    try:
        processor = AutoProcessor.from_pretrained(
            "microsoft/git-base",
            token=token,
            cache_dir="./model_cache",
            trust_remote_code=True
        )
        print("✅ Processor loaded successfully")
    except Exception as e:
        print(f"❌ Processor loading failed: {e}")
        return
    
    # Step 4: Test model loading
    print("\n4. Testing model loading...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/git-base",
            token=token,
            cache_dir="./model_cache",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Step 5: Test EscalatorVLMAnalyzer
    print("\n5. Testing EscalatorVLMAnalyzer...")
    try:
        from escalator_vlm_analyzer import EscalatorVLMAnalyzer
        analyzer = EscalatorVLMAnalyzer(hf_token=token)
        print(f"✅ Analyzer created successfully")
        print(f"   VLM Available: {analyzer.vlm_available}")
        print(f"   Processor: {analyzer.vlm_processor is not None}")
        print(f"   Model: {analyzer.vlm_model is not None}")
    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}")
        return
    
    print("\n=== Test completed successfully ===")

if __name__ == "__main__":
    test_vlm_loading() 