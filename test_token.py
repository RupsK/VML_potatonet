#!/usr/bin/env python3
"""
Test Hugging Face Token
Simple script to test if the token works for model loading
"""

import os
import sys
from transformers import AutoProcessor, AutoModelForVision2Seq

def test_token(token):
    """Test if the token works for model loading"""
    print("🧪 Testing Hugging Face Token...")
    print("=" * 50)
    
    if not token:
        print("❌ No token provided!")
        return False
    
    if not token.startswith("hf_"):
        print("❌ Token doesn't start with 'hf_' - invalid format!")
        return False
    
    print(f"✅ Token format looks valid: {token[:10]}...")
    
    # Test with a simple model
    model_name = "microsoft/git-base"
    
    try:
        print(f"\n🔄 Testing token with {model_name}...")
        
        # Test processor loading
        processor = AutoProcessor.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("✅ Processor loaded successfully!")
        
        # Test model loading
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully!")
        
        print(f"\n🎉 Token is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False

if __name__ == "__main__":
    # Use the provided token
    token = "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"
    
    if not token:
        print("Usage: python test_token.py <your_token>")
        print("Or set HF_TOKEN environment variable")
        sys.exit(1)
    
    success = test_token(token)
    if success:
        print("\n✅ Token test completed successfully!")
    else:
        print("\n❌ Token test failed!")
        sys.exit(1) 