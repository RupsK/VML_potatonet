#!/usr/bin/env python3
"""
Test Different Models
Test that different models load correctly and produce different outputs
"""

import os
import sys
from thermal_vlm_processor import ThermalImageProcessor

def test_different_models():
    """Test different models to ensure they load correctly"""
    print("🧪 Testing Different Models")
    print("=" * 50)
    
    # Test different models
    models = {
        "BLIP Base": "Salesforce/blip-image-captioning-base",
        "BLIP Large": "Salesforce/blip-image-captioning-large", 
        "GIT Base": "microsoft/git-base"
    }
    
    test_image = "test_image/1.jpeg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n🔄 Testing {model_name}...")
        
        try:
            # Initialize processor
            processor = ThermalImageProcessor()
            
            # Load specific model
            processor.load_model(model_path)
            
            if not processor.model_loaded:
                print(f"❌ Failed to load {model_name}")
                continue
            
            print(f"✅ {model_name} loaded successfully!")
            
            # Test analysis
            result = processor.analyze_thermal_image(
                test_image,
                prompt="Describe this thermal image briefly."
            )
            
            if result:
                results[model_name] = result
                print(f"✅ {model_name} analysis completed!")
                print(f"   Caption: {result['caption'][:100]}...")
            else:
                print(f"❌ {model_name} analysis failed!")
                
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    # Compare results
    print(f"\n📊 Results Summary:")
    print("=" * 50)
    
    if len(results) > 1:
        print("✅ Multiple models working!")
        
        # Check if outputs are different
        captions = [result['caption'] for result in results.values()]
        unique_captions = set(captions)
        
        if len(unique_captions) == len(captions):
            print("✅ Different models produce different outputs!")
        else:
            print("⚠️ Some models produce similar outputs")
            
        for model_name, result in results.items():
            print(f"\n🤖 {model_name}:")
            print(f"   Model: {result['model']}")
            print(f"   Time: {result['processing_time']:.2f}s")
            print(f"   Caption: {result['caption'][:150]}...")
    else:
        print("❌ Only one or no models working")
        return False
    
    return True

if __name__ == "__main__":
    success = test_different_models()
    if success:
        print("\n🎉 All models working correctly!")
    else:
        print("\n❌ Some models failed!")
        sys.exit(1) 