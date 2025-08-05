# streamlit_escalator_vlm.py - VLM-Enhanced Escalator Safety Monitor
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from escalator_vlm_analyzer import EscalatorVLMAnalyzer
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="VLM-Enhanced Escalator Safety Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for VLM-enhanced alerts
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #9C27B0, #673AB7);
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
        border-left: 4px solid #9C27B0;
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
    
    .vlm-enhanced {
        background-color: #f3e5f5;
        border: 2px solid #9c27b0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #4a148c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_escalator_vlm_analyzer():
    """Load VLM-enhanced escalator analyzer"""
    # Get Hugging Face token - PRIORITY ORDER for security:
    hf_token = None
    
    # 1. FIRST: Try Streamlit secrets (for production deployment)
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
        if hf_token:
            st.success("🔐 Token loaded from Streamlit secrets (production)")
    except Exception:
        pass
    
    # 2. SECOND: Try session state (for user input)
    if not hf_token:
        hf_token = st.session_state.get("hf_token", None)
        if hf_token:
            st.info("🔑 Token loaded from session state")
    
    # 3. THIRD: Try environment variable (for local development)
    if not hf_token:
        import os
        hf_token = os.getenv("HF_TOKEN", None)
        if hf_token:
            st.info("🔑 Token loaded from environment variable")
    
    # 4. LAST: Try token file (for local development only)
    if not hf_token:
        try:
            with open("hf_token.txt", "r") as f:
                hf_token = f.read().strip()
                st.warning("⚠️ Token loaded from file (not recommended for production)")
        except FileNotFoundError:
            pass
    
    # Debug token status
    if hf_token:
        st.info(f"🔑 Token found: {hf_token[:10]}...")
    else:
        st.warning("🔑 No Hugging Face token found - VLM will not work")
        st.info("💡 To add your token securely:")
        st.info("   • For Streamlit Cloud: Add to app secrets")
        st.info("   • For local: Set HF_TOKEN environment variable")
    
    return EscalatorVLMAnalyzer(hf_token=hf_token)

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #9C27B0; font-size: 3rem; margin-bottom: 10px;">🤖 VLM-Enhanced Escalator Safety Monitor</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">AI-powered detection of crowding and falling objects using Vision-Language Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #9C27B0, #673AB7); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0; text-align: center;">⚙️ VLM Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Hugging Face Token input
    st.sidebar.markdown("**🤖 VLM Configuration:**")
    hf_token = st.sidebar.text_input(
        "Hugging Face Token (for VLM):",
        type="password",
        help="Enter your Hugging Face token to enable AI-enhanced analysis",
        placeholder="hf_..."
    )
    
    if hf_token:
        st.session_state.hf_token = hf_token
        st.sidebar.success("✅ Token saved for VLM analysis!")
    
    # Analysis settings
    max_frames = st.sidebar.slider(
        "Max Frames to Analyze:",
        min_value=5,
        max_value=20,
        value=10,
        help="More frames = more accurate VLM analysis"
    )
    
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode:",
        ["enhanced", "detailed"],
        index=0,
        help="Enhanced: VLM + basic analysis, Detailed: Full VLM analysis"
    )
    
    # Safety thresholds
    st.sidebar.markdown("**🚨 Safety Thresholds:**")
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
    <div style="background: linear-gradient(90deg, #9C27B0, #673AB7); padding: 15px; border-radius: 15px; margin: 30px 0 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">📤 Upload Escalator Video for AI Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an escalator video file",
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
        help="Upload escalator video for VLM-enhanced safety analysis"
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
                
                # Show video preview
                st.markdown("**📹 Video Preview:**")
                try:
                    with open(test_file_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes, start_time=0)
                    st.caption(f"**{selected_test_file}** - Click 'Analyze with VLM' to start AI analysis")
                except Exception as e:
                    st.warning(f"Could not preview video: {e}")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🤖 Analyze with VLM", use_container_width=True):
                        analyze_escalator_vlm(test_file_path, analysis_mode, max_frames, crowding_threshold, falling_threshold)
    
    # Handle uploaded file
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show video preview
        st.markdown("**📹 Uploaded Video Preview:**")
        try:
            st.video(uploaded_file, start_time=0)
            st.caption(f"**{uploaded_file.name}** - Click 'Analyze with VLM' to start AI analysis")
        except Exception as e:
            st.warning(f"Could not preview video: {e}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🤖 Analyze Uploaded Video with VLM", use_container_width=True):
                analyze_escalator_vlm(temp_path, analysis_mode, max_frames, crowding_threshold, falling_threshold)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_escalator_vlm(video_path, analysis_mode, max_frames, crowding_threshold, falling_threshold):
    """Analyze escalator video with VLM enhancement"""
    
    try:
        # Get token with better error handling
        hf_token = None
        
        # Try multiple sources for token
        if os.path.exists("hf_token.txt"):
            try:
                with open("hf_token.txt", "r") as f:
                    hf_token = f.read().strip()
                st.info(f"🔑 Token loaded from file: {hf_token[:10]}...")
            except Exception as e:
                st.warning(f"Could not read token file: {e}")
        
        # Also try session state
        if not hf_token:
            hf_token = st.session_state.get("hf_token", None)
            if hf_token:
                st.info(f"🔑 Token loaded from session: {hf_token[:10]}...")
        
        if not hf_token:
            st.error("❌ No Hugging Face token found! Please enter your token in the sidebar.")
            return
        
        with st.spinner("🤖 Loading VLM-enhanced escalator analyzer..."):
            # Force reload analyzer each time with explicit token
            st.info("Creating EscalatorVLMAnalyzer...")
            st.info(f"Token being passed: {hf_token[:10]}...")
            
            # Create analyzer with explicit token
            analyzer = EscalatorVLMAnalyzer(hf_token=hf_token)
            
            # Wait a moment for initialization
            import time
            time.sleep(2)
            
            # Debug: Check what happened during initialization
            st.info(f"Analyzer created. VLM status: {analyzer.vlm_available}")
            st.info(f"Processor loaded: {analyzer.vlm_processor is not None}")
            st.info(f"Model loaded: {analyzer.vlm_model is not None}")
            
            if not analyzer.vlm_available:
                st.warning("VLM failed to initialize. This might be due to Streamlit environment constraints.")
                st.info("Check the terminal output for detailed error messages.")
        
        # Debug VLM status with more details
        st.info(f"🔍 VLM Status: Available={analyzer.vlm_available}, Processor={analyzer.vlm_processor is not None}, Model={analyzer.vlm_model is not None}")
        
        if not analyzer.vlm_available:
            st.error("❌ VLM failed to initialize. Please check your token and try again.")
            return
        
        # Update analyzer settings
        analyzer.max_frames_to_analyze = max_frames
        
        with st.spinner(f"🤖 Analyzing escalator safety with VLM {analysis_mode} mode..."):
            result = analyzer.analyze_escalator_vlm(video_path, analysis_mode)
        
        if result and 'error' not in result:
            display_vlm_results(result, crowding_threshold, falling_threshold, video_path)
            st.success("✅ VLM-enhanced escalator safety analysis completed!")
        else:
            error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
            st.error(f"VLM analysis failed: {error_msg}")
    
    except Exception as e:
        st.error(f"Error during VLM analysis: {str(e)}")
        st.info("💡 Try using a smaller video file or reducing frame count.")

def display_vlm_results(result, crowding_threshold, falling_threshold, video_path):
    """Display VLM-enhanced escalator safety analysis results"""
    
    # VLM status
    if result.get('vlm_used'):
        st.markdown(f'<div class="vlm-enhanced">🤖 AI Enhanced: VLM model successfully loaded and used for analysis</div>', unsafe_allow_html=True)
    else:
        st.warning("WARNING: VLM not available - using basic analysis only")
    
    # Safety alerts section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #9C27B0, #673AB7); padding: 15px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🚨 AI-Enhanced Safety Alerts</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display safety alerts
    safety_alerts = result['safety_alerts']
    for alert in safety_alerts:
        if "HIGH" in alert or "CRITICAL" in alert:
            st.markdown(f'<div class="alert-high">🚨 {alert}</div>', unsafe_allow_html=True)
        elif "MODERATE" in alert or "WARNING:" in alert:
            st.markdown(f'<div class="alert-moderate">⚠️ {alert}</div>', unsafe_allow_html=True)
        elif "AI Enhanced" in alert:
            st.markdown(f'<div class="vlm-enhanced">🤖 {alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-safe">✅ {alert}</div>', unsafe_allow_html=True)
    
    # Video display and information
    st.subheader("📹 Video Analysis")
    
    # Display the video
    try:
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        
        st.video(video_bytes, start_time=0)
        video_info = result.get('video_info', {})
        st.caption(f"📹 **{video_info.get('file_name', 'Video')}** - Duration: {video_info.get('duration', 0):.2f}s | Resolution: {video_info.get('resolution', 'Unknown')}")
        
    except Exception as e:
        st.warning(f"Could not display video: {e}")
    
    # Video metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{video_info.get('duration', 0):.2f}s")
    with col2:
        st.metric("Resolution", video_info.get('resolution', 'Unknown'))
    with col3:
        st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
    with col4:
        st.metric("Frames Analyzed", result['total_frames_analyzed'])
    
    # VLM-enhanced crowding analysis
    crowding_analysis = result['crowding_analysis']
    st.subheader("👥 VLM-Enhanced Crowding Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crowding_status = "🚨 HIGH" if crowding_analysis.get('crowding_detected') and crowding_analysis['crowding_level'] == 'high' else \
                         "WARNING: MODERATE" if crowding_analysis.get('crowding_detected') and crowding_analysis['crowding_level'] == 'moderate' else \
                         "✅ LOW"
        st.metric("Crowding Level", crowding_status)
    with col2:
        st.metric("Crowding Score", f"{crowding_analysis.get('average_crowding_score', 0):.1f}/100")
    with col3:
        st.metric("VLM Enhanced", "✅ Yes" if crowding_analysis.get('vlm_enhanced') else "❌ No")
    with col4:
        st.metric("Crowding %", f"{crowding_analysis.get('crowding_percentage', 0):.1f}%")
    
    # VLM-enhanced falling object analysis
    falling_analysis = result['falling_analysis']
    st.subheader("📦 VLM-Enhanced Falling Object Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        falling_status = "🚨 HIGH" if falling_analysis.get('falling_detected') and falling_analysis['falling_risk'] == 'high' else \
                        "WARNING: MODERATE" if falling_analysis.get('falling_detected') and falling_analysis['falling_risk'] == 'moderate' else \
                        "✅ LOW"
        st.metric("Falling Risk", falling_status)
    with col2:
        st.metric("Risk Score", f"{falling_analysis.get('average_falling_risk', 0):.1f}/100")
    with col3:
        st.metric("VLM Enhanced", "✅ Yes" if falling_analysis.get('vlm_enhanced') else "❌ No")
    with col4:
        st.metric("Risk %", f"{falling_analysis.get('falling_risk_percentage', 0):.1f}%")
    
    # VLM-enhanced safety summary
    st.subheader("📋 AI-Enhanced Safety Summary")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #9C27B0;
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
    
    # Frame-by-frame VLM analysis
    if result['frame_analyses']:
        st.subheader("🎞️ Frame-by-Frame VLM Analysis")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 VLM Safety Table", "📈 VLM Trends", "🤖 AI Insights"])
        
        with tab1:
            # VLM safety summary table
            summary_data = []
            for analysis in result['frame_analyses']:
                summary_data.append({
                    "Frame": analysis['frame_number'],
                    "Time (s)": f"{analysis['timestamp']:.2f}",
                    "Crowding Score": f"{analysis['crowding_score']:.1f}",
                    "VLM Crowding": f"{analysis.get('crowding_vlm_score', 0):.1f}",
                    "Falling Risk": f"{analysis['falling_detection']['falling_probability']:.1f}",
                    "VLM Falling": f"{analysis.get('falling_vlm_score', 0):.1f}",
                    "VLM Status": "✅" if analysis.get('vlm_description') and 'VLM analysis failed' not in analysis.get('vlm_description', '') else "❌"
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            # VLM trends graph
            if len(result['frame_analyses']) > 1:
                timestamps = [analysis['timestamp'] for analysis in result['frame_analyses']]
                crowding_scores = [analysis['crowding_score'] for analysis in result['frame_analyses']]
                vlm_crowding_scores = [analysis.get('crowding_vlm_score', 0) for analysis in result['frame_analyses']]
                falling_risks = [analysis['falling_detection']['falling_probability'] for analysis in result['frame_analyses']]
                vlm_falling_scores = [analysis.get('falling_vlm_score', 0) for analysis in result['frame_analyses']]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Crowding comparison
                ax1.plot(timestamps, crowding_scores, 'o-', color='#FF5722', linewidth=2, markersize=6, label='Basic Crowding')
                ax1.plot(timestamps, vlm_crowding_scores, 's-', color='#9C27B0', linewidth=2, markersize=6, label='VLM Crowding')
                ax1.axhline(y=crowding_threshold, color='red', linestyle='--', alpha=0.7, label=f'Alert Threshold ({crowding_threshold})')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Crowding Score')
                ax1.set_title('VLM vs Basic Crowding Detection')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.set_facecolor('#f8f9fa')
                
                # Falling risk comparison
                ax2.plot(timestamps, falling_risks, 'o-', color='#FF9800', linewidth=2, markersize=6, label='Basic Falling Risk')
                ax2.plot(timestamps, vlm_falling_scores, 's-', color='#673AB7', linewidth=2, markersize=6, label='VLM Falling Risk')
                ax2.axhline(y=falling_threshold, color='purple', linestyle='--', alpha=0.7, label=f'Alert Threshold ({falling_threshold})')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Falling Risk Score')
                ax2.set_title('VLM vs Basic Falling Risk Detection')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            # Show VLM AI insights with frame images
            for i, analysis in enumerate(result['frame_analyses'][:6]):  # Show first 6 frames
                with st.expander(f"Frame {analysis['frame_number']} (t={analysis['timestamp']:.2f}s) - AI Analysis"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        # Display frame image if available
                        try:
                            # Extract frame from video
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, analysis['frame_number'])
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.image(frame_rgb, caption=f"Frame {analysis['frame_number']}", use_column_width=True)
                            else:
                                st.info("Frame image not available")
                        except Exception as e:
                            st.info(f"Could not display frame: {e}")
                    
                    with col2:
                        st.markdown(f"**Basic Crowding:** {analysis['crowding_score']:.1f}/100")
                        st.markdown(f"**VLM Crowding:** {analysis.get('crowding_vlm_score', 0):.1f}/100")
                        st.markdown(f"**Basic Falling Risk:** {analysis['falling_detection']['falling_probability']:.1f}/100")
                        st.markdown(f"**VLM Falling Risk:** {analysis.get('falling_vlm_score', 0):.1f}/100")
                        
                        st.markdown("**🤖 AI Description:**")
                        vlm_desc = analysis.get('vlm_description', 'No VLM analysis available')
                        if 'VLM analysis failed' not in vlm_desc:
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #f3e5f5; 
                                    color: #4a148c; 
                                    padding: 10px; 
                                    border-radius: 5px; 
                                    font-size: 12px;
                                    line-height: 1.4;
                                    border-left: 3px solid #9c27b0;
                                ">
                                {vlm_desc}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("VLM analysis failed for this frame")
    
    # Processing information
    st.subheader("⚙️ Processing Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analysis Mode", result['analysis_mode'].title())
    with col2:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    with col3:
        st.metric("VLM Used", "✅ Yes" if result.get('vlm_used') else "❌ No")
    with col4:
        st.metric("Frames per Second", f"{result['total_frames_analyzed']/result['processing_time']:.1f}")

if __name__ == "__main__":
    main() 