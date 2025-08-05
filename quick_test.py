#!/usr/bin/env python3
"""
Quick test for the ensemble system
"""

from thermal_vlm_ensemble import ThermalVLMEnsemble
import os

def test_ensemble():
    """Quick test of the ensemble system"""
    print("🧪 Quick Ensemble Test")
    print("=" * 30)
    
    # Initialize ensemble
    print("🔧 Initializing ensemble...")
    ensemble = ThermalVLMEnsemble()
    
    # Test image
    test_image = "test_image/download.jpg"
    
    print("📊 Running ensemble analysis...")
    
    # Test all ensemble methods
    methods = ["weighted_average", "majority_vote", "best_model"]
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"🎯 {method.upper().replace('_', ' ')} METHOD:")
        print(f"{'='*50}")
        
        result = ensemble.analyze_with_ensemble(test_image, ensemble_method=method)
        
        if result:
            print(f"✅ {method} analysis successful!")
            print(f"   Processing time: {result['total_processing_time']:.2f}s")
            
            # Print temperature analysis info
            temp_analysis = result['temperature_analysis']
            print(f"   Temperature analysis keys: {list(temp_analysis.keys())}")
            print(f"   Mean temperature: {temp_analysis['mean_temperature']:.1f}")
            print(f"   Max temperature: {temp_analysis['max_temperature']:.1f}")
            print(f"   Min temperature: {temp_analysis['min_temperature']:.1f}")
            
            print(f"\n{'='*50}")
            print("🎯 FULL ENSEMBLE CAPTION:")
            print(f"{'='*50}")
            print(result['ensemble_caption'])
            print(f"{'='*50}")
            
            # Print individual model results
            print("\n📊 Individual Model Results:")
            print()
            
            for model_name, model_result in result['individual_results'].items():
                print(f"🔹 {model_name}:")
                print(f"   Confidence: {model_result['confidence']:.2f}")
                print(f"   Caption: {model_result['caption']}")
                print()
        else:
            print(f"❌ {method} analysis failed!")

if __name__ == "__main__":
    test_ensemble() 