#!/usr/bin/env python3
"""
Test script for safety incident detection
Specifically tests the unattended suitcase scenario and similar safety incidents
"""

from escalator_vlm_analyzer import EscalatorVLMAnalyzer

def test_safety_incident_detection():
    """Test the improved safety incident detection"""
    print("Testing Safety Incident Detection")
    print("=" * 50)
    
    # Create analyzer instance
    analyzer = EscalatorVLMAnalyzer()
    
    # Test cases based on the interface scenario
    test_cases = [
        # Frame 0 scenario - normal escalator usage
        "The escalator is seen in this undated image with people waiting",
        
        # Frame 12 scenario - safety incident
        "This is the terrifying moment when a woman was knocked over by an unattended suitcase",
        
        # Additional safety incident scenarios
        "A person was knocked down the escalator by falling luggage",
        "Unattended suitcase caused safety concern on escalator",
        "Woman knocked over by unattended bag on escalator",
        "Safety incident: person fell due to unattended luggage",
        "Terrifying moment as luggage knocked someone over",
        
        # Normal scenarios for comparison
        "People waiting on escalator",
        "Normal escalator operation with passengers",
        "Escalator with light traffic"
    ]
    
    print("Safety Incident Detection Results:")
    print("-" * 50)
    
    for i, original in enumerate(test_cases, 1):
        improved = analyzer._simplify_vlm_description(original)
        
        # Test falling risk detection
        falling_score = analyzer._extract_falling_from_vlm([improved])
        
        # Test safety keyword detection
        safety_keywords = ['fall', 'falling', 'crowd', 'crowded', 'safety', 'risk', 'concern', 
                          'knocked', 'unattended', 'incident', 'terrifying', 'suitcase', 'luggage']
        has_safety_concerns = any(keyword in improved.lower() for keyword in safety_keywords)
        
        # Determine incident level
        if falling_score > 60:
            incident_level = "🔴 HIGH RISK"
        elif falling_score > 30:
            incident_level = "🟡 MODERATE RISK"
        else:
            incident_level = "🟢 LOW RISK"
        
        print(f"Test {i}: {incident_level}")
        print(f"  Original: {original}")
        print(f"  Improved: {improved}")
        print(f"  Falling Risk Score: {falling_score:.1f}/100")
        print(f"  Safety Concerns: {'⚠️ YES' if has_safety_concerns else '✅ NO'}")
        print()

def test_interface_scenario():
    """Test the specific scenario from the interface"""
    print("Testing Interface Scenario")
    print("=" * 40)
    
    analyzer = EscalatorVLMAnalyzer()
    
    # Frame 0 scenario
    frame0_text = "The escalator is seen in this undated image with people waiting"
    frame0_improved = analyzer._simplify_vlm_description(frame0_text)
    frame0_crowding = analyzer._extract_crowding_from_vlm([frame0_improved])
    frame0_falling = analyzer._extract_falling_from_vlm([frame0_improved])
    
    print("Frame 0 (t=0.00s) - Normal Operation:")
    print(f"  Description: {frame0_improved}")
    print(f"  Crowding Score: {frame0_crowding:.1f}/100")
    print(f"  Falling Risk: {frame0_falling:.1f}/100")
    print(f"  Status: {'🟢 NORMAL' if frame0_falling < 30 else '⚠️ CONCERN'}")
    print()
    
    # Frame 12 scenario - the safety incident
    frame12_text = "This is the terrifying moment when a woman was knocked over by an unattended suitcase"
    frame12_improved = analyzer._simplify_vlm_description(frame12_text)
    frame12_crowding = analyzer._extract_crowding_from_vlm([frame12_improved])
    frame12_falling = analyzer._extract_falling_from_vlm([frame12_improved])
    
    print("Frame 12 (t=0.48s) - Safety Incident:")
    print(f"  Description: {frame12_improved}")
    print(f"  Crowding Score: {frame12_crowding:.1f}/100")
    print(f"  Falling Risk: {frame12_falling:.1f}/100")
    print(f"  Status: {'🔴 HIGH RISK' if frame12_falling > 60 else '🟡 MODERATE RISK'}")
    print()
    
    # Overall assessment
    print("Overall Safety Assessment:")
    print("-" * 30)
    if frame12_falling > 60:
        print("🔴 CRITICAL: High falling risk detected in Frame 12")
        print("   → Unattended luggage causing safety incident")
        print("   → Immediate attention required")
    elif frame12_falling > 30:
        print("🟡 WARNING: Moderate safety concern in Frame 12")
        print("   → Potential falling object risk")
        print("   → Monitor situation closely")
    else:
        print("🟢 NORMAL: No significant safety concerns detected")
    
    # Compare with interface metrics
    print("\nInterface Metrics Comparison:")
    print("-" * 35)
    print("Frame 0 - Interface shows:")
    print("  • Basic Crowding: 86.3/100")
    print("  • VLM Crowding: 0.0/100")
    print("  • Our Analysis: {:.1f}/100".format(frame0_crowding))
    print()
    print("Frame 12 - Interface shows:")
    print("  • Basic Crowding: 100.0/100")
    print("  • VLM Crowding: 0.0/100")
    print("  • Our Analysis: {:.1f}/100".format(frame12_crowding))
    print("  • Falling Risk: {:.1f}/100".format(frame12_falling))

def test_ai_description_generation():
    """Test AI description generation for the interface"""
    print("\n" + "=" * 50)
    print("AI Description Generation for Interface")
    print("=" * 50)
    
    analyzer = EscalatorVLMAnalyzer()
    
    # Test cases that might be generated by the VLM
    vlm_outputs = [
        "The escalator is seen in this undated image with people waiting",
        "This is the terrifying moment when a woman was knocked over by an unattended suitcase",
        "A person on escalator with potential safety concern",
        "Unattended luggage causing safety incident on escalator"
    ]
    
    print("VLM Output Processing for Interface:")
    print("-" * 45)
    
    for i, vlm_output in enumerate(vlm_outputs):
        improved = analyzer._simplify_vlm_description(vlm_output)
        print(f"VLM Output {i+1}:")
        print(f"  Raw: {vlm_output}")
        print(f"  Processed: {improved}")
        print(f"  Suitable for interface: {'✅ YES' if len(improved) <= 120 else '❌ TOO LONG'}")
        print()

if __name__ == "__main__":
    print("Safety Incident Detection Testing")
    print("This script tests the improved safety incident detection capabilities")
    print("including the unattended suitcase scenario from the interface")
    print()
    
    # Test general safety incident detection
    test_safety_incident_detection()
    
    # Test the specific interface scenario
    test_interface_scenario()
    
    # Test AI description generation
    test_ai_description_generation()
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("The system now better handles:")
    print("• Unattended luggage detection")
    print("• Safety incident identification")
    print("• Falling object risk assessment")
    print("• Interface-compatible descriptions") 