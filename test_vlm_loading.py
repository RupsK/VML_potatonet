#!/usr/bin/env python3
"""
Test VLM model loading for escalator analysis
"""
import time
import os

def test_vlm_loading():
    print("🔍 Testing VLM model loading...")
    
    # Check if token file exists
    if os.path.exists("hf_token.txt"):
        with open("hf_token.txt", "r") as f:
            hf_token = f.read().strip()
        print(f"✅ Hugging Face token found: {hf_token[:10]}...")
    else:
        print("❌ No Hugging Face token found")
        return False
    
    try:
        print("🔄 Importing transformers...")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        print("✅ Transformers imported successfully")
        
        print("🔄 Loading VLM model (microsoft/git-base)...")
        start_time = time.time()
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            "microsoft/git-base",
            token=hf_token,
            cache_dir="./model_cache"
        )
        print(f"✅ Processor loaded in {time.time() - start_time:.2f}s")
        
        # Load model
        start_time = time.time()
        model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/git-base",
            token=hf_token,
            cache_dir="./model_cache",
            torch_dtype=torch.float32
        )
        print(f"✅ Model loaded in {time.time() - start_time:.2f}s")
        
        print("🎉 VLM model loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ VLM model loading test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_vlm_loading() 