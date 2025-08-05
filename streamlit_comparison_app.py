import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from thermal_vlm_comparison import ThermalVLMComparison
import os
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Thermal Image VLM Comparison",
    page_icon="🔥",
    layout="wide"
)

# Initialize the comparison processor
@st.cache_resource
def load_comparison_processor():
    """Load the thermal VLM comparison processor with caching"""
    return ThermalVLMComparison()

def main():
    st.title("🔥 Thermal Image VLM Model Comparison")
    st.markdown("Compare BLIP vs LLaVA-Next analysis of thermal images")
    
    # Sidebar for options
    st.sidebar.header("Settings")
    
    # Custom prompt
    custom_prompt = st.sidebar.text_area(
        "Custom Analysis Prompt (optional):",
        value="Analyze this thermal image in detail, focusing on temperature patterns and visible objects.",
        height=100
    )
    
    # Main content area
    st.header("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a thermal image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a thermal image to compare BLIP vs LLaVA-Next analysis"
    )
    
    # Or select from test images
    st.subheader("📁 Or Select from Test Images")
    test_folder = "test_image"
    if os.path.exists(test_folder):
        test_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            test_images.extend(Path(test_folder).glob(f"*{ext}"))
            test_images.extend(Path(test_folder).glob(f"*{ext.upper()}"))
        
        if test_images:
            selected_test_image = st.selectbox(
                "Choose from test images:",
                [img.name for img in test_images],
                index=0 if test_images else None
            )
            
            if selected_test_image:
                test_image_path = str(Path(test_folder) / selected_test_image)
                if st.button("Compare Models on Selected Image"):
                    compare_models(test_image_path, custom_prompt)
        else:
            st.info("No test images found in the test_image folder")
    else:
        st.info("Test image folder not found")
    
    # Handle uploaded file
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Compare models on uploaded image
        compare_models(temp_path, custom_prompt)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def compare_models(image_path, custom_prompt):
    """Compare BLIP and LLaVA-Next analysis of a thermal image"""
    
    try:
        with st.spinner("Loading VLM models..."):
            processor = load_comparison_processor()
        
        with st.spinner("Comparing model analysis..."):
            result = processor.compare_models(image_path, custom_prompt)
        
        if result:
            # Display the thermal image
            st.subheader("📸 Thermal Image")
            st.image(result['image'], caption="Processed Thermal Image", use_column_width=True)
            
            # Display temperature analysis
            st.subheader("🌡️ Temperature Analysis")
            temp_data = result['temperature_analysis']
            
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Temp", f"{temp_data['mean_temperature']:.1f}")
            with col2:
                st.metric("Max Temp", f"{temp_data['max_temperature']:.1f}")
            with col3:
                st.metric("Min Temp", f"{temp_data['min_temperature']:.1f}")
            with col4:
                st.metric("Temp Range", f"{temp_data['temperature_range']:.1f}")
            
            # Model comparison section
            st.subheader("🤖 VLM Model Comparison")
            
            # Create two columns for side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔥 BLIP Analysis")
                st.markdown(f"**Model:** {result['blip_result']['model']}")
                st.markdown(f"**Processing Time:** {result['blip_result']['processing_time']:.2f}s")
                st.markdown(f"**Caption Length:** {result['comparison']['blip_caption_length']} characters")
                
                # Display BLIP caption
                st.markdown("**AI Description:**")
                st.info(result['blip_result']['caption'])
            
            with col2:
                st.markdown("### 🚀 LLaVA-Next Analysis")
                st.markdown(f"**Model:** {result['llava_result']['model']}")
                st.markdown(f"**Processing Time:** {result['llava_result']['processing_time']:.2f}s")
                st.markdown(f"**Caption Length:** {result['comparison']['llava_caption_length']} characters")
                
                # Display LLaVA-Next caption
                st.markdown("**AI Description:**")
                st.success(result['llava_result']['caption'])
            
            # Comparison metrics
            st.subheader("📊 Comparison Metrics")
            
            # Create comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Processing time comparison
            models = ['BLIP', 'LLaVA-Next']
            times = [result['comparison']['blip_processing_time'], result['comparison']['llava_processing_time']]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars1 = ax1.bar(models, times, color=colors, alpha=0.7)
            ax1.set_ylabel('Processing Time (seconds)')
            ax1.set_title('Processing Speed Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time_val in zip(bars1, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{time_val:.2f}s', ha='center', va='bottom')
            
            # Caption length comparison
            lengths = [result['comparison']['blip_caption_length'], result['comparison']['llava_caption_length']]
            
            bars2 = ax2.bar(models, lengths, color=colors, alpha=0.7)
            ax2.set_ylabel('Caption Length (characters)')
            ax2.set_title('Description Detail Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, length_val in zip(bars2, lengths):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{length_val}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed comparison table
            st.subheader("📋 Detailed Comparison")
            
            comparison_data = {
                'Metric': ['Processing Time', 'Caption Length', 'Model Type', 'Description Style'],
                'BLIP': [
                    f"{result['comparison']['blip_processing_time']:.2f}s",
                    f"{result['comparison']['blip_caption_length']} chars",
                    'Vision-Language Model',
                    'Direct image captioning'
                ],
                'LLaVA-Next': [
                    f"{result['comparison']['llava_processing_time']:.2f}s",
                    f"{result['comparison']['llava_caption_length']} chars",
                    'Large Language Vision Assistant',
                    'Conversational analysis'
                ]
            }
            
            st.table(comparison_data)
            
            # Analysis insights
            st.subheader("💡 Analysis Insights")
            
            # Determine which model performed better
            blip_time = result['comparison']['blip_processing_time']
            llava_time = result['comparison']['llava_processing_time']
            blip_length = result['comparison']['blip_caption_length']
            llava_length = result['comparison']['llava_caption_length']
            
            insights = []
            
            if blip_time < llava_time:
                insights.append("🔥 **BLIP is faster** - Better for real-time applications")
            else:
                insights.append("🚀 **LLaVA-Next is faster** - More efficient processing")
            
            if blip_length < llava_length:
                insights.append("🚀 **LLaVA-Next provides more detailed descriptions** - Better for comprehensive analysis")
            else:
                insights.append("🔥 **BLIP provides concise descriptions** - Good for quick overview")
            
            # Check for repetitive patterns in BLIP output
            if "thermal thermal" in result['blip_result']['caption'].lower():
                insights.append("⚠️ **BLIP shows repetitive patterns** - Common issue with thermal images")
            
            if "struggled" in result['blip_result']['caption'].lower():
                insights.append("❌ **BLIP struggled with this thermal image** - LLaVA-Next may be more robust")
            
            for insight in insights:
                st.markdown(insight)
        
        else:
            st.error("Failed to compare models. Please try again.")
    
    except Exception as e:
        st.error(f"Error during model comparison: {str(e)}")
        st.info("Make sure you have the required dependencies installed.")

if __name__ == "__main__":
    main() 