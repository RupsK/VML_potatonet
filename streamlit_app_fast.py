# streamlit_app_fast.py - Fast version prioritizing simple processing
import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from thermal_video_processor_simple import SimpleThermalVideoProcessor
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Thermal Video Analyzer - Fast Mode",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
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
    
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_simple_video_processor():
    """Load simple video processor with caching"""
    return SimpleThermalVideoProcessor()

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #4CAF50; font-size: 3rem; margin-bottom: 10px;">⚡ Fast Thermal Video Analyzer</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">Quick video analysis with optimized processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0; text-align: center;">⚙️ Fast Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis settings
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode:",
        ["summary", "key_frames"],
        index=0,
        help="Summary: Quick overview, Key Frames: Scene changes"
    )
    
    max_frames = st.sidebar.slider(
        "Max Frames:",
        min_value=3,
        max_value=15,
        value=5,
        help="More frames = more detailed but slower"
    )
    
    # Custom prompt
    custom_prompt = st.sidebar.text_area(
        "Analysis Prompt:",
        value="Analyze this thermal video. Describe temperature patterns, objects, and any changes over time.",
        height=80
    )
    
    # Main content
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); padding: 15px; border-radius: 15px; margin: 30px 0 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🎬 Upload Thermal Video</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a thermal video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a thermal video to analyze"
    )
    
    # Test videos section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f093fb, #f5576c); padding: 12px; border-radius: 12px; margin: 25px 0 15px 0;">
        <h3 style="color: white; margin: 0; text-align: center;">📁 Or Select from Test Videos</h3>
    </div>
    """, unsafe_allow_html=True)
    
    test_folder = "test_video"
    if os.path.exists(test_folder):
        test_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            test_files.extend(Path(test_folder).glob(f"*{ext}"))
            test_files.extend(Path(test_folder).glob(f"*{ext.upper()}"))
        
        if test_files:
            selected_test_file = st.selectbox(
                "Choose from test videos:",
                [vid.name for vid in test_files],
                index=0 if test_files else None
            )
            
            if selected_test_file:
                test_file_path = str(Path(test_folder) / selected_test_file)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("⚡ Analyze Selected Test Video", use_container_width=True):
                        analyze_video_fast(test_file_path, custom_prompt, analysis_mode, max_frames)
    
    # Handle uploaded file
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        analyze_video_fast(temp_path, custom_prompt, analysis_mode, max_frames)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_video_fast(video_path, custom_prompt, analysis_mode, max_frames):
    """Fast video analysis using simple processor"""
    
    try:
        with st.spinner("⚡ Loading fast video processor..."):
            processor = load_simple_video_processor()
        
        # Update processor settings
        processor.max_frames_to_analyze = max_frames
        
        with st.spinner(f"⚡ Analyzing video with {analysis_mode} mode..."):
            result = processor.process_video(video_path, custom_prompt, analysis_mode)
        
        if result and 'error' not in result:
            display_fast_results(result)
            st.success("✅ Fast analysis completed!")
        else:
            error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
            st.error(f"Analysis failed: {error_msg}")
    
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("💡 Try using a smaller video file or reducing frame count.")

def display_fast_results(result):
    """Display fast analysis results"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); padding: 15px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">⚡ Fast Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Video information
    video_info = result['video_info']
    st.subheader("📹 Video Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{video_info.get('duration', 0):.2f}s")
    with col2:
        st.metric("Resolution", video_info.get('resolution', 'Unknown'))
    with col3:
        st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
    with col4:
        st.metric("Frames Analyzed", result['total_frames_analyzed'])
    
    # Video summary
    st.subheader("📋 Video Summary")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #4CAF50;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        ">
        {result['video_summary']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Frame-by-frame analysis
    if result['frame_analyses']:
        st.subheader("🎞️ Frame Analysis")
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Summary Table", "🎯 Frame Details"])
        
        with tab1:
            # Summary table
            summary_data = []
            for analysis in result['frame_analyses']:
                summary_data.append({
                    "Frame": analysis['frame_number'],
                    "Time (s)": f"{analysis['timestamp']:.2f}",
                    "Mean Temp": f"{analysis['temperature_analysis']['mean_temperature']:.1f}",
                    "Description": analysis['caption'][:80] + "..." if len(analysis['caption']) > 80 else analysis['caption']
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            # Show frame details
            for i, analysis in enumerate(result['frame_analyses'][:8]):  # Show first 8 frames
                with st.expander(f"Frame {analysis['frame_number']} (t={analysis['timestamp']:.2f}s)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Temperature:** {analysis['temperature_analysis']['mean_temperature']:.1f}")
                        st.markdown(f"**Max Temp:** {analysis['temperature_analysis']['max_temperature']:.1f}")
                        st.markdown(f"**Min Temp:** {analysis['temperature_analysis']['min_temperature']:.1f}")
                    with col2:
                        st.markdown("**AI Description:**")
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #2d2d2d; 
                                color: #e0e0e0; 
                                padding: 10px; 
                                border-radius: 5px; 
                                font-size: 12px;
                                line-height: 1.4;
                            ">
                            {analysis['caption']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
    # Processing information
    st.subheader("⚙️ Processing Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analysis Mode", result['analysis_mode'].title())
    with col2:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    with col3:
        st.metric("Frames per Second", f"{result['total_frames_analyzed']/result['processing_time']:.1f}")

if __name__ == "__main__":
    main() 