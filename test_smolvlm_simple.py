#!/usr/bin/env python3
"""
Simple SmolVLM Test
Test SmolVLM without secrets file dependency
"""

import os
import sys
from thermal_smolvlm_processor import SmolVLMProcessor

def test_smolvlm_simple():
    """Test SmolVLM processor with token from file"""
    print("🧪 Simple SmolVLM Test (No Secrets File)")
    print("=" * 50)
    
    # Read token from file
    try:
        with open("hf_token.txt", "r") as f:
            token = f.read().strip()
        print(f"✅ Token loaded from file: {token[:10]}...")
    except FileNotFoundError:
        print("❌ Token file not found!")
        return False
    
    # Initialize processor
    processor = SmolVLMProcessor(hf_token=token)
    
    # Test model loading
    print("\n🔄 Loading model...")
    processor.load_model()
    
    if not processor.model_loaded:
        print("❌ Model loading failed!")
        return False
    
    print("✅ Model loaded successfully!")
    
    # Test with a sample image
    test_image = "test_image/1.jpeg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"\n🔄 Analyzing image: {test_image}")
    
    # Test analysis
    result = processor.analyze_thermal_image(
        test_image,
        prompt="Describe this thermal image briefly."
    )
    
    if result:
        print("✅ Analysis completed successfully!")
        print(f"🤖 Model: {result['model']}")
        print(f"⏱️ Time: {result['processing_time']:.2f}s")
        print(f"📝 Caption: {result['caption'][:100]}...")
        return True
    else:
        print("❌ Analysis failed!")
        return False

if __name__ == "__main__":
    success = test_smolvlm_simple()
    if success:
        print("\n🎉 SmolVLM works without secrets file!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 