#!/usr/bin/env python3
"""
Test the VLM description simplification function
"""

from escalator_vlm_analyzer import EscalatorVLMAnalyzer

def test_description_simplification():
    """Test the _simplify_vlm_description function"""
    
    # Create analyzer instance
    analyzer = EscalatorVLMAnalyzer()
    
    # Test cases
    test_cases = [
        "passengers wait on the escalator at the metro station",
        "passengers on escalators at the metro station",
        "passengers on the escalator at the metro station",
        "people walking on escalator with bag fall down",
        "crowd of people on escalator with luggage",
        "escalator with people and falling objects",
        "people on escalator at subway station",
        "busy escalator with many passengers"
    ]
    
    print("🧪 Testing VLM Description Simplification")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        simplified = analyzer._simplify_vlm_description(test_case)
        print(f"\n{i}. Original: {test_case}")
        print(f"   Simplified: {simplified}")
    
    print("\n✅ Description simplification test completed!")

if __name__ == "__main__":
    test_description_simplification() 