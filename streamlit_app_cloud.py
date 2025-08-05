# streamlit_app_cloud.py - Streamlit Cloud Compatible Version
import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Try to import seaborn, but don't fail if it's not available
SEABORN_AVAILABLE = False
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    # seaborn not available, will use matplotlib only
    pass

import pandas as pd
import os
from pathlib import Path

# Safe OpenCV import for Streamlit Cloud
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV not available: {e}")
    OPENCV_AVAILABLE = False

# Safe imports with error handling
try:
    from thermal_vlm_processor import ThermalImageProcessor
    THERMAL_VLM_AVAILABLE = True
except ImportError as e:
    st.warning(f"Thermal VLM Processor not available: {e}")
    THERMAL_VLM_AVAILABLE = False

try:
    from thermal_smolvlm_processor import SmolVLMProcessor
    SMOLVLM_AVAILABLE = True
except ImportError as e:
    st.warning(f"SmolVLM Processor not available: {e}")
    SMOLVLM_AVAILABLE = False

try:
    from thermal_video_processor import ThermalVideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    st.warning(f"Video Processor not available: {e}")
    VIDEO_PROCESSOR_AVAILABLE = False

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Thermal Image AI Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Custom styling for better appearance */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Enhanced metric styling */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        border-left: 4px solid #ff6b35;
    }
    
    /* Better file uploader styling */
    .stFileUploader {
        border: 2px dashed #ff6b35;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Better text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff6b35;
        box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize processors with error handling
@st.cache_resource
def load_processor():
    """Load the thermal image processor with caching"""
    if not THERMAL_VLM_AVAILABLE:
        st.error("Thermal VLM Processor not available")
        return None
    try:
        return ThermalImageProcessor()
    except Exception as e:
        st.error(f"Failed to load Thermal VLM Processor: {e}")
        return None

@st.cache_resource
def load_smolvlm_processor():
    """Load the SmolVLM processor with caching"""
    if not SMOLVLM_AVAILABLE:
        st.error("SmolVLM Processor not available")
        return None
    try:
        return SmolVLMProcessor()
    except Exception as e:
        st.error(f"Failed to load SmolVLM Processor: {e}")
        return None

@st.cache_resource
def load_video_processor():
    """Load the video processor with caching"""
    if not VIDEO_PROCESSOR_AVAILABLE:
        st.error("Video Processor not available")
        return None
    try:
        return ThermalVideoProcessor()
    except Exception as e:
        st.error(f"Failed to load Video Processor: {e}")
        return None

def main():
    # Enhanced header with better styling
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #ff6b35, #f7931e); border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 3rem; margin-bottom: 10px;">🔥 Thermal Image AI Analyzer</h1>
        <p style="color: white; font-size: 1.2rem; margin: 0;">Advanced AI-powered thermal image and video analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # System status check
    st.sidebar.markdown("### 🔧 System Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    
    with status_col1:
        if OPENCV_AVAILABLE:
            st.success("✅ OpenCV")
        else:
            st.error("❌ OpenCV")
            
        if THERMAL_VLM_AVAILABLE:
            st.success("✅ Thermal VLM")
        else:
            st.error("❌ Thermal VLM")
    
    with status_col2:
        if SMOLVLM_AVAILABLE:
            st.success("✅ SmolVLM")
        else:
            st.error("❌ SmolVLM")
            
        if VIDEO_PROCESSOR_AVAILABLE:
            st.success("✅ Video Processor")
        else:
            st.error("❌ Video Processor")

    # Main content
    st.markdown("## 📊 Analysis Options")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis", "📈 Results"])
    
    with tab1:
        st.markdown("### Image Analysis")
        
        if not THERMAL_VLM_AVAILABLE:
            st.error("Image analysis is not available. Please check the system status.")
        else:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a thermal image for analysis"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Analysis options
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Basic Analysis", "VLM Analysis", "SmolVLM Analysis"]
                )
                
                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        try:
                            processor = load_processor()
                            if processor:
                                # Perform analysis based on type
                                if analysis_type == "Basic Analysis":
                                    st.success("Basic analysis completed!")
                                    st.info("This is a placeholder for basic analysis")
                                elif analysis_type == "VLM Analysis":
                                    st.success("VLM analysis completed!")
                                    st.info("This is a placeholder for VLM analysis")
                                elif analysis_type == "SmolVLM Analysis":
                                    st.success("SmolVLM analysis completed!")
                                    st.info("This is a placeholder for SmolVLM analysis")
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
    
    with tab2:
        st.markdown("### Video Analysis")
        
        if not VIDEO_PROCESSOR_AVAILABLE:
            st.error("Video analysis is not available. Please check the system status.")
        else:
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov'],
                help="Upload a video for analysis"
            )
            
            if uploaded_video is not None:
                st.video(uploaded_video)
                
                if st.button("🎬 Analyze Video"):
                    with st.spinner("Analyzing video..."):
                        try:
                            processor = load_video_processor()
                            if processor:
                                st.success("Video analysis completed!")
                                st.info("This is a placeholder for video analysis")
                        except Exception as e:
                            st.error(f"Video analysis failed: {e}")
    
    with tab3:
        st.markdown("### Analysis Results")
        st.info("Results will appear here after analysis is completed.")
        
        # Placeholder for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Temperature", "25°C", "2°C")
        
        with col2:
            st.metric("Anomalies", "3", "1")
        
        with col3:
            st.metric("Confidence", "85%", "5%")

if __name__ == "__main__":
    main() 