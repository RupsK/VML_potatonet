#!/usr/bin/env python3
"""
Test BLIP Base Fix
Quick test to verify BLIP Base works after the image key fix
"""

import os
import sys
from thermal_vlm_processor import ThermalImageProcessor

def test_blip_fix():
    """Test BLIP Base to ensure it works correctly"""
    print("🧪 Testing BLIP Base Fix")
    print("=" * 40)
    
    test_image = "test_image/1.jpeg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    try:
        # Initialize processor
        processor = ThermalImageProcessor()
        
        # Load BLIP Base model
        print("🔄 Loading BLIP Base model...")
        processor.load_model("Salesforce/blip-image-captioning-base")
        
        if not processor.model_loaded:
            print("❌ Failed to load BLIP Base model")
            return False
        
        print("✅ BLIP Base model loaded successfully!")
        
        # Test analysis
        print("🔄 Analyzing image...")
        result = processor.analyze_thermal_image(
            test_image,
            prompt="Describe this thermal image briefly."
        )
        
        if result:
            print("✅ Analysis completed successfully!")
            print(f"🤖 Model: {result['model']}")
            print(f"⏱️ Time: {result['processing_time']:.2f}s")
            print(f"📝 Caption: {result['caption'][:100]}...")
            
            # Check result keys
            print(f"\n📋 Result keys: {list(result.keys())}")
            
            # Check if enhanced_image is present
            if 'enhanced_image' in result:
                print("✅ 'enhanced_image' key found!")
            else:
                print("❌ 'enhanced_image' key missing!")
            
            return True
        else:
            print("❌ Analysis failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_blip_fix()
    if success:
        print("\n🎉 BLIP Base fix successful!")
    else:
        print("\n❌ BLIP Base fix failed!")
        sys.exit(1) 