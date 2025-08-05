#!/usr/bin/env python3
"""
Debug script for SmolVLM model loading
"""

from thermal_smolvlm_processor import SmolVLMProcessor
import time

def debug_smolvlm():
    """Debug SmolVLM model loading"""
    print("🔍 Debugging SmolVLM Model Loading")
    print("=" * 50)
    
    # Initialize processor
    print("1. Initializing SmolVLM processor...")
    processor = SmolVLMProcessor()
    print(f"   ✅ Processor initialized")
    print(f"   Device: {processor.device}")
    print(f"   Model loaded: {processor.model_loaded}")
    
    # Try to load model
    print("\n2. Attempting to load SmolVLM model...")
    try:
        processor.load_model()
        print(f"   Model loaded: {processor.model_loaded}")
        
        if processor.model_loaded:
            print("   ✅ SmolVLM model loaded successfully!")
            print(f"   Model: {processor.model}")
            print(f"   Processor: {processor.processor}")
        else:
            print("   ❌ SmolVLM model failed to load")
            
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with a simple image
    print("\n3. Testing with sample image...")
    test_image = "test_image/1.jpeg"
    
    try:
        result = processor.analyze_thermal_image(
            test_image,
            prompt="Describe this thermal image."
        )
        
        if result:
            print("   ✅ Analysis completed!")
            print(f"   Model used: {result['model']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Caption: {result['caption'][:100]}...")
            
            if "fallback" in result['model'].lower():
                print("   ⚠️ Using fallback analysis - model not working properly")
            else:
                print("   ✅ Using actual SmolVLM model!")
        else:
            print("   ❌ Analysis failed")
            
    except Exception as e:
        print(f"   ❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_smolvlm() 