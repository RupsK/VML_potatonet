# streamlit_general_video.py - General Video Analyzer
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from general_video_analyzer import GeneralVideoAnalyzer
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="General Video Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #2196F3, #1976D2);
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
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_general_analyzer():
    """Load general video analyzer with caching"""
    return GeneralVideoAnalyzer()

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #2196F3; font-size: 3rem; margin-bottom: 10px;">🎬 General Video Analyzer</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">Analyze regular videos for motion, scene changes, and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #2196F3, #1976D2); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0; text-align: center;">⚙️ Analysis Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis settings
    max_frames = st.sidebar.slider(
        "Max Frames to Analyze:",
        min_value=3,
        max_value=20,
        value=8,
        help="More frames = more detailed analysis but slower processing"
    )
    
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode:",
        ["summary", "detailed"],
        index=0,
        help="Summary: Quick overview, Detailed: In-depth analysis"
    )
    
    # Main content
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2196F3, #1976D2); padding: 15px; border-radius: 15px; margin: 30px 0 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">📤 Upload Video</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
        help="Upload a video to analyze"
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
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
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
                    if st.button("🎬 Analyze Selected Test Video", use_container_width=True):
                        analyze_video(test_file_path, analysis_mode, max_frames)
    
    # Handle uploaded file
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        analyze_video(temp_path, analysis_mode, max_frames)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_video(video_path, analysis_mode, max_frames):
    """Analyze video using general analyzer"""
    
    try:
        with st.spinner("🎬 Loading video analyzer..."):
            analyzer = load_general_analyzer()
        
        # Update analyzer settings
        analyzer.max_frames_to_analyze = max_frames
        
        with st.spinner(f"🎬 Analyzing video with {analysis_mode} mode..."):
            result = analyzer.analyze_video(video_path, analysis_mode)
        
        if result and 'error' not in result:
            display_general_results(result)
            st.success("✅ Video analysis completed!")
        else:
            error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
            st.error(f"Analysis failed: {error_msg}")
    
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("💡 Try using a smaller video file or reducing frame count.")

def display_general_results(result):
    """Display general video analysis results"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2196F3, #1976D2); padding: 15px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🎬 Video Analysis Results</h2>
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
    
    # Motion analysis
    motion_analysis = result['motion_analysis']
    st.subheader("🏃 Motion Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        motion_status = "✅ Yes" if motion_analysis.get('motion_detected') else "❌ No"
        st.metric("Motion Detected", motion_status)
    with col2:
        st.metric("Motion Intensity", f"{motion_analysis.get('motion_percentage', 0):.1f}%")
    with col3:
        st.metric("Avg Motion Score", f"{motion_analysis.get('average_motion_score', 0):.2f}")
    with col4:
        st.metric("Motion Variance", f"{motion_analysis.get('motion_variance', 0):.2f}")
    
    # Scene analysis
    scene_analysis = result['scene_analysis']
    st.subheader("🎭 Scene Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Scene Changes", scene_analysis.get('scene_changes', 0))
    with col2:
        st.metric("Consistency", f"{scene_analysis.get('consistency_score', 1.0):.2f}")
    with col3:
        st.metric("Brightness Trend", scene_analysis.get('brightness_trend', 'Unknown'))
    
    # Video summary
    st.subheader("📋 Analysis Summary")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #2196F3;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 400px;
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
        tab1, tab2, tab3 = st.tabs(["📊 Summary Table", "📈 Brightness Graph", "🎯 Frame Details"])
        
        with tab1:
            # Summary table
            summary_data = []
            for analysis in result['frame_analyses']:
                summary_data.append({
                    "Frame": analysis['frame_number'],
                    "Time (s)": f"{analysis['timestamp']:.2f}",
                    "Brightness": f"{analysis['brightness']:.1f}",
                    "Contrast": f"{analysis['contrast']:.1f}",
                    "Motion Score": f"{analysis['motion_score']:.2f}",
                    "Description": analysis['description'][:60] + "..." if len(analysis['description']) > 60 else analysis['description']
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            # Brightness graph
            if len(result['frame_analyses']) > 1:
                timestamps = [analysis['timestamp'] for analysis in result['frame_analyses']]
                brightness_values = [analysis['brightness'] for analysis in result['frame_analyses']]
                motion_scores = [analysis['motion_score'] for analysis in result['frame_analyses']]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Brightness over time
                ax1.plot(timestamps, brightness_values, 'o-', color='#2196F3', linewidth=2, markersize=6)
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Brightness')
                ax1.set_title('Brightness Variation Over Time')
                ax1.grid(True, alpha=0.3)
                ax1.set_facecolor('#f8f9fa')
                
                # Motion scores over time
                ax2.plot(timestamps, motion_scores, 'o-', color='#FF5722', linewidth=2, markersize=6)
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Motion Score')
                ax2.set_title('Motion Detection Over Time')
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            # Show frame details
            for i, analysis in enumerate(result['frame_analyses'][:6]):  # Show first 6 frames
                with st.expander(f"Frame {analysis['frame_number']} (t={analysis['timestamp']:.2f}s)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Brightness:** {analysis['brightness']:.1f}")
                        st.markdown(f"**Contrast:** {analysis['contrast']:.1f}")
                        st.markdown(f"**Motion Score:** {analysis['motion_score']:.2f}")
                    with col2:
                        st.markdown("**Frame Description:**")
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
                            {analysis['description']}
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