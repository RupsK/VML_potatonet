#!/usr/bin/env python3
"""
Demo script for Thermal Image AI Analyzer
This script demonstrates how to use the thermal image processor with sample images
"""

import os
from pathlib import Path
from thermal_vlm_processor import ThermalImageProcessor
import matplotlib.pyplot as plt
import numpy as np

def demo_thermal_analysis():
    """Demonstrate thermal image analysis with sample images"""
    
    print("🔥 Thermal Image AI Analyzer - Demo")
    print("=" * 50)
    
    # Initialize the processor
    print("Initializing thermal image processor...")
    processor = ThermalImageProcessor()
    
    # Check if test images exist
    test_folder = "test_image"
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found!")
        print("Please ensure you have sample thermal images in the test_image folder.")
        return
    
    # Get all test images
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        test_images.extend(Path(test_folder).glob(f"*{ext}"))
        test_images.extend(Path(test_folder).glob(f"*{ext.upper()}"))
    
    if not test_images:
        print(f"❌ No images found in '{test_folder}' folder!")
        print("Please add some thermal images to the test_image folder.")
        return
    
    print(f"✅ Found {len(test_images)} test images")
    
    # Process each image
    results = []
    for i, img_path in enumerate(test_images, 1):
        print(f"\n📸 Processing image {i}/{len(test_images)}: {img_path.name}")
        print("-" * 40)
        
        # Analyze the image
        result = processor.analyze_thermal_image(str(img_path))
        
        if result:
            result['filename'] = img_path.name
            results.append(result)
            
            # Display results
            print(f"🔍 AI Description: {result['caption']}")
            print("\n🌡️ Temperature Analysis:")
            temp_data = result['temperature_analysis']
            for key, value in temp_data.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"❌ Failed to analyze {img_path.name}")
    
    # Summary
    print(f"\n🎉 Demo completed! Successfully analyzed {len(results)} images.")
    
    # Save results
    if results:
        output_file = "demo_analysis_results.txt"
        processor.save_analysis_results(results, output_file)
        print(f"📄 Results saved to: {output_file}")
    
    # Show comparison
    if len(results) > 1:
        show_comparison(results)

def show_comparison(results):
    """Show a comparison of temperature statistics across images"""
    
    print("\n📊 Comparison of Temperature Statistics:")
    print("=" * 50)
    
    # Extract temperature data
    filenames = [r['filename'] for r in results]
    mean_temps = [r['temperature_analysis']['mean_temperature'] for r in results]
    max_temps = [r['temperature_analysis']['max_temperature'] for r in results]
    min_temps = [r['temperature_analysis']['min_temperature'] for r in results]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temperature range comparison
    x = np.arange(len(filenames))
    width = 0.35
    
    ax1.bar(x - width/2, min_temps, width, label='Min Temperature', color='blue', alpha=0.7)
    ax1.bar(x + width/2, max_temps, width, label='Max Temperature', color='red', alpha=0.7)
    ax1.set_xlabel('Images')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature Range Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.split('.')[0] for f in filenames], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean temperature comparison
    bars = ax2.bar(x, mean_temps, color='green', alpha=0.7)
    ax2.set_xlabel('Images')
    ax2.set_ylabel('Mean Temperature')
    ax2.set_title('Mean Temperature Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.split('.')[0] for f in filenames], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_temps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('temperature_comparison.png', dpi=300, bbox_inches='tight')
    print("📈 Comparison chart saved as: temperature_comparison.png")
    
    # Show the plot
    plt.show()

def demo_custom_prompts():
    """Demonstrate custom prompt analysis"""
    
    print("\n🎯 Custom Prompt Analysis Demo")
    print("=" * 40)
    
    # Initialize processor
    processor = ThermalImageProcessor()
    
    # Test image
    test_image = "test_image/1.jpeg"
    if not os.path.exists(test_image):
        print(f"❌ Test image {test_image} not found!")
        return
    
    # Custom prompts for different analysis types
    custom_prompts = [
        "Focus on detecting hot spots and areas of high temperature in this thermal image",
        "Analyze the temperature distribution and identify any thermal anomalies",
        "Describe the objects and their thermal signatures visible in this image",
        "Identify any potential safety concerns or unusual thermal patterns"
    ]
    
    print(f"Analyzing {test_image} with different custom prompts...")
    
    for i, prompt in enumerate(custom_prompts, 1):
        print(f"\n🔍 Prompt {i}: {prompt}")
        print("-" * 30)
        
        result = processor.analyze_thermal_image(test_image, prompt)
        if result:
            print(f"AI Response: {result['caption']}")
        else:
            print("❌ Analysis failed")

if __name__ == "__main__":
    # Run the main demo
    demo_thermal_analysis()
    
    # Run custom prompt demo
    demo_custom_prompts()
    
    print("\n✨ Demo completed! Check the generated files for detailed results.")
    print("\nTo run the interactive web interface, use:")
    print("streamlit run streamlit_app.py") 