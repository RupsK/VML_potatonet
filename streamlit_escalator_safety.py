# streamlit_escalator_safety.py - Escalator Safety Monitor
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from escalator_analyzer import EscalatorAnalyzer
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Escalator Safety Monitor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for safety alerts
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #FF5722, #E64A19);
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
        border-left: 4px solid #FF5722;
    }
    
    .alert-high {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #c62828;
        font-weight: bold;
    }
    
    .alert-moderate {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #ef6c00;
        font-weight: bold;
    }
    
    .alert-safe {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #2e7d32;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_escalator_analyzer():
    """Load escalator safety analyzer with caching"""
    return EscalatorAnalyzer()

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #FF5722; font-size: 3rem; margin-bottom: 10px;">🚇 Escalator Safety Monitor</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">Detect crowding and falling objects for escalator safety</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #FF5722, #E64A19); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0; text-align: center;">⚙️ Safety Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis settings
    max_frames = st.sidebar.slider(
        "Max Frames to Analyze:",
        min_value=5,
        max_value=20,
        value=12,
        help="More frames = more accurate safety detection"
    )
    
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode:",
        ["safety", "detailed"],
        index=0,
        help="Safety: Quick safety check, Detailed: In-depth analysis"
    )
    
    # Safety thresholds
    st.sidebar.markdown("**Safety Thresholds:**")
    crowding_threshold = st.sidebar.slider(
        "Crowding Alert Threshold:",
        min_value=30,
        max_value=80,
        value=50,
        help="Higher = more sensitive to crowding"
    )
    
    falling_threshold = st.sidebar.slider(
        "Falling Risk Threshold:",
        min_value=20,
        max_value=70,
        value=40,
        help="Higher = more sensitive to falling objects"
    )
    
    # Main content
    st.markdown("""
    <div style="background: linear-gradient(90deg, #FF5722, #E64A19); padding: 15px; border-radius: 15px; margin: 30px 0 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">📤 Upload Escalator Video</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an escalator video file",
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
        help="Upload escalator video for safety analysis"
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
                    if st.button("🚇 Analyze Escalator Safety", use_container_width=True):
                        analyze_escalator_safety(test_file_path, analysis_mode, max_frames, crowding_threshold, falling_threshold)
    
    # Handle uploaded file
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        analyze_escalator_safety(temp_path, analysis_mode, max_frames, crowding_threshold, falling_threshold)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_escalator_safety(video_path, analysis_mode, max_frames, crowding_threshold, falling_threshold):
    """Analyze escalator video for safety concerns"""
    
    try:
        with st.spinner("🚇 Loading escalator safety analyzer..."):
            analyzer = load_escalator_analyzer()
        
        # Update analyzer settings
        analyzer.max_frames_to_analyze = max_frames
        
        with st.spinner(f"🚇 Analyzing escalator safety with {analysis_mode} mode..."):
            result = analyzer.analyze_escalator_safety(video_path, analysis_mode)
        
        if result and 'error' not in result:
            display_safety_results(result, crowding_threshold, falling_threshold)
            st.success("✅ Escalator safety analysis completed!")
        else:
            error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
            st.error(f"Safety analysis failed: {error_msg}")
    
    except Exception as e:
        st.error(f"Error during safety analysis: {str(e)}")
        st.info("💡 Try using a smaller video file or reducing frame count.")

def display_safety_results(result, crowding_threshold, falling_threshold):
    """Display escalator safety analysis results"""
    
    # Safety alerts section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #FF5722, #E64A19); padding: 15px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🚨 Safety Alerts</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display safety alerts
    safety_alerts = result['safety_alerts']
    for alert in safety_alerts:
        if "HIGH" in alert or "CRITICAL" in alert:
            st.markdown(f'<div class="alert-high">🚨 {alert}</div>', unsafe_allow_html=True)
        elif "MODERATE" in alert or "⚠️" in alert:
            st.markdown(f'<div class="alert-moderate">⚠️ {alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-safe">✅ {alert}</div>', unsafe_allow_html=True)
    
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
    
    # Crowding analysis
    crowding_analysis = result['crowding_analysis']
    st.subheader("👥 Crowding Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crowding_status = "🚨 HIGH" if crowding_analysis.get('crowding_detected') and crowding_analysis['crowding_level'] == 'high' else \
                         "⚠️ MODERATE" if crowding_analysis.get('crowding_detected') and crowding_analysis['crowding_level'] == 'moderate' else \
                         "✅ LOW"
        st.metric("Crowding Level", crowding_status)
    with col2:
        st.metric("Crowding Score", f"{crowding_analysis.get('average_crowding_score', 0):.1f}/100")
    with col3:
        st.metric("Max Crowding", f"{crowding_analysis.get('max_crowding_score', 0):.1f}/100")
    with col4:
        st.metric("Crowding Trend", crowding_analysis.get('crowding_trend', 'Unknown'))
    
    # Falling object analysis
    falling_analysis = result['falling_analysis']
    st.subheader("📦 Falling Object Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        falling_status = "🚨 HIGH" if falling_analysis.get('falling_detected') and falling_analysis['falling_risk'] == 'high' else \
                        "⚠️ MODERATE" if falling_analysis.get('falling_detected') and falling_analysis['falling_risk'] == 'moderate' else \
                        "✅ LOW"
        st.metric("Falling Risk", falling_status)
    with col2:
        st.metric("Risk Score", f"{falling_analysis.get('average_falling_risk', 0):.1f}/100")
    with col3:
        st.metric("Max Risk", f"{falling_analysis.get('max_falling_risk', 0):.1f}/100")
    with col4:
        st.metric("Potential Objects", falling_analysis.get('total_potential_objects', 0))
    
    # Safety summary
    st.subheader("📋 Safety Summary")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #FF5722;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        ">
        {result['safety_summary']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Frame-by-frame analysis
    if result['frame_analyses']:
        st.subheader("🎞️ Frame-by-Frame Safety Analysis")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Safety Table", "📈 Safety Trends", "🎯 Frame Details"])
        
        with tab1:
            # Safety summary table
            summary_data = []
            for analysis in result['frame_analyses']:
                summary_data.append({
                    "Frame": analysis['frame_number'],
                    "Time (s)": f"{analysis['timestamp']:.2f}",
                    "Crowding Score": f"{analysis['crowding_score']:.1f}",
                    "Falling Risk": f"{analysis['falling_detection']['falling_probability']:.1f}",
                    "Objects Detected": analysis['falling_detection']['potential_objects'],
                    "Safety Status": "🚨" if analysis['crowding_score'] > crowding_threshold or analysis['falling_detection']['falling_probability'] > falling_threshold else "✅"
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            # Safety trends graph
            if len(result['frame_analyses']) > 1:
                timestamps = [analysis['timestamp'] for analysis in result['frame_analyses']]
                crowding_scores = [analysis['crowding_score'] for analysis in result['frame_analyses']]
                falling_risks = [analysis['falling_detection']['falling_probability'] for analysis in result['frame_analyses']]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Crowding over time
                ax1.plot(timestamps, crowding_scores, 'o-', color='#FF5722', linewidth=2, markersize=6, label='Crowding Score')
                ax1.axhline(y=crowding_threshold, color='red', linestyle='--', alpha=0.7, label=f'Alert Threshold ({crowding_threshold})')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Crowding Score')
                ax1.set_title('Crowding Detection Over Time')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.set_facecolor('#f8f9fa')
                
                # Falling risk over time
                ax2.plot(timestamps, falling_risks, 'o-', color='#9C27B0', linewidth=2, markersize=6, label='Falling Risk')
                ax2.axhline(y=falling_threshold, color='purple', linestyle='--', alpha=0.7, label=f'Alert Threshold ({falling_threshold})')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Falling Risk Score')
                ax2.set_title('Falling Object Risk Over Time')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            # Show frame details
            for i, analysis in enumerate(result['frame_analyses'][:8]):  # Show first 8 frames
                with st.expander(f"Frame {analysis['frame_number']} (t={analysis['timestamp']:.2f}s)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Crowding Score:** {analysis['crowding_score']:.1f}/100")
                        st.markdown(f"**Falling Risk:** {analysis['falling_detection']['falling_probability']:.1f}/100")
                        st.markdown(f"**Objects Detected:** {analysis['falling_detection']['potential_objects']}")
                        st.markdown(f"**Brightness:** {analysis['brightness']:.1f}")
                    with col2:
                        st.markdown("**Safety Description:**")
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
                            {analysis['safety_description']}
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