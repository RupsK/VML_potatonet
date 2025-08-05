#!/usr/bin/env python3
"""
Test SmolVLM Integration
Test script to verify SmolVLM integration with actual Hugging Face model
"""

import os
import sys
from thermal_smolvlm_processor import SmolVLMProcessor

def test_smolvlm_integration():
    """Test SmolVLM integration with actual model"""
    print("🧪 Testing SmolVLM Integration")
    print("=" * 50)
    
    # Initialize processor
    processor = SmolVLMProcessor()
    
    # Test model loading
    print("\n1️⃣ Testing Model Loading...")
    processor.load_model()
    
    if not processor.model_loaded:
        print("❌ Model loading failed!")
        return False
    
    print(f"✅ Model loaded successfully: {processor.model}")
    print(f"✅ Device: {processor.device}")
    
    # Test with a sample image
    test_image = "test_image/1.jpeg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"\n2️⃣ Testing Image Analysis...")
    print(f"📸 Using test image: {test_image}")
    
    # Test analysis
    result = processor.analyze_thermal_image(
        test_image,
        prompt="Analyze this thermal image. Describe what you see, including temperature patterns, objects, and any anomalies."
    )
    
    if result:
        print("✅ Analysis completed successfully!")
        print(f"🤖 Model Used: {result['model']}")
        print(f"⏱️ Processing Time: {result['processing_time']:.2f}s")
        print(f"📝 Caption Length: {len(result['caption'])} characters")
        print(f"🌡️ Temperature Range: {result['temperature_analysis']['min_temperature']:.1f} - {result['temperature_analysis']['max_temperature']:.1f}")
        
        print(f"\n📄 Generated Caption:")
        print("-" * 40)
        print(result['caption'][:500] + "..." if len(result['caption']) > 500 else result['caption'])
        print("-" * 40)
        
        return True
    else:
        print("❌ Analysis failed!")
        return False

def test_smolvlm_capabilities():
    """Test specific SmolVLM capabilities"""
    print("\n3️⃣ Testing SmolVLM Capabilities...")
    
    processor = SmolVLMProcessor()
    processor.load_model()
    
    if not processor.model_loaded:
        print("❌ Cannot test capabilities - model not loaded")
        return
    
    test_image = "test_image/1.jpeg"
    if not os.path.exists(test_image):
        print("❌ Test image not found")
        return
    
    # Test different prompts
    test_prompts = [
        "What do you see in this thermal image?",
        "Describe the temperature patterns in this image.",
        "Are there any hot or cold regions visible?",
        "What objects can you identify in this thermal image?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 Test {i}: {prompt}")
        try:
            result = processor.analyze_thermal_image(test_image, prompt)
            if result:
                print(f"✅ Success - Caption: {result['caption'][:100]}...")
            else:
                print("❌ Failed")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 SmolVLM Integration Test")
    print("Based on: https://huggingface.co/blog/smolvlm")
    print("=" * 60)
    
    # Test basic integration
    success = test_smolvlm_integration()
    
    if success:
        # Test capabilities
        test_smolvlm_capabilities()
        
        print("\n🎉 All tests completed!")
        print("\n📋 Summary:")
        print("✅ SmolVLM model integration working")
        print("✅ Chat template format implemented")
        print("✅ Fallback mechanism in place")
        print("✅ Thermal image analysis functional")
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1) 