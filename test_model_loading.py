#!/usr/bin/env python3
"""
Test Model Loading
Simple script to test if the vision-language model loads correctly
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

def test_model_loading(hf_token=None):
    """Test different model loading approaches"""
    print("🧪 Testing Model Loading...")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if hf_token:
        print(f"Using Hugging Face token: {hf_token[:10]}...")
    else:
        print("No Hugging Face token provided")
    
    # Create cache directory
    os.makedirs("./model_cache", exist_ok=True)
    
    # Test different model variants
    model_variants = [
        ("microsoft/git-base", "GIT Base"),
        ("microsoft/git-base-coco", "GIT Base COCO"),
        ("microsoft/git-large", "GIT Large")
    ]
    
    for model_name, model_display in model_variants:
        print(f"\n🔄 Testing {model_display}...")
        try:
            # Load processor
            processor_kwargs = {
                "trust_remote_code": True,
                "cache_dir": "./model_cache"
            }
            if hf_token:
                processor_kwargs["token"] = hf_token
            
            processor = AutoProcessor.from_pretrained(
                model_name,
                **processor_kwargs
            )
            print(f"✅ Processor loaded for {model_display}")
            
            # Load model
            model_kwargs = {
                "torch_dtype": torch.float32,
                "device_map": None,
                "trust_remote_code": True,
                "cache_dir": "./model_cache"
            }
            
            if hf_token:
                model_kwargs["token"] = hf_token
            
            if device == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                })
            
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if device == "cpu":
                model = model.to(device)
            
            model.eval()
            print(f"✅ Model loaded successfully: {model_display}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Has chat template: {hasattr(processor, 'apply_chat_template')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_display}: {e}")
            continue
    
    print("\n❌ All model variants failed to load")
    return False

if __name__ == "__main__":
    # Use the provided token
    token = "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"
    success = test_model_loading(token)
    if success:
        print("\n✅ Model loading test completed successfully!")
    else:
        print("\n❌ Model loading test failed!") 