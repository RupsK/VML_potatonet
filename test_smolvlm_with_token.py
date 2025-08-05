#!/usr/bin/env python3
"""
Test SmolVLM with Token
Test the SmolVLM processor with the user's Hugging Face token
"""

import os
import sys
from thermal_smolvlm_processor import SmolVLMProcessor

def test_smolvlm_with_token():
    """Test SmolVLM processor with token"""
    print("🧪 Testing SmolVLM Processor with Token...")
    print("=" * 60)
    
    # Use the provided token
    token = "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"
    
    # Initialize processor with token
    processor = SmolVLMProcessor(hf_token=token)
    
    # Test model loading
    print("\n1️⃣ Testing Model Loading...")
    processor.load_model()
    
    if not processor.model_loaded:
        print("❌ Model loading failed!")
        return False
    
    print(f"✅ Model loaded successfully!")
    print(f"✅ Device: {processor.device}")
    print(f"✅ Model type: {type(processor.model).__name__}")
    print(f"✅ Has chat template: {hasattr(processor.processor, 'apply_chat_template')}")
    
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
        
        if result['temperature_analysis']:
            temp = result['temperature_analysis']
            print(f"🌡️ Temperature Range: {temp['min_temperature']:.1f} - {temp['max_temperature']:.1f}")
            print(f"🌡️ Mean Temperature: {temp['mean_temperature']:.1f}")
            print(f"🔥 Hot Regions: {temp['hot_regions_percentage']:.1f}%")
            print(f"❄️ Cold Regions: {temp['cold_regions_percentage']:.1f}%")
            print(f"👤 Human Patterns: {temp['potential_human_patterns']}")
        
        print(f"\n📄 Generated Caption:")
        print("-" * 50)
        print(result['caption'][:500] + "..." if len(result['caption']) > 500 else result['caption'])
        print("-" * 50)
        
        return True
    else:
        print("❌ Analysis failed!")
        return False

if __name__ == "__main__":
    print("🚀 SmolVLM Token Integration Test")
    print("=" * 60)
    
    success = test_smolvlm_with_token()
    if success:
        print("\n🎉 All tests passed! SmolVLM is working with your token!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 