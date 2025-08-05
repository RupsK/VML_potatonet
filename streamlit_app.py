# streamlit_app.py
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
from thermal_vlm_processor import ThermalImageProcessor
from thermal_vlm_comparison import ThermalVLMComparison
from thermal_vlm_ensemble import ThermalVLMEnsemble
from thermal_knowledge_injection import ThermalKnowledgeInjector
from thermal_smolvlm_processor import SmolVLMProcessor
from thermal_video_processor import ThermalVideoProcessor
from thermal_video_processor_simple import SimpleThermalVideoProcessor
from general_video_processor import GeneralVideoProcessor
import os
from pathlib import Path

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

# Initialize the thermal processor
@st.cache_resource
def load_processor():
    """Load the thermal image processor with caching"""
    return ThermalImageProcessor()

# Initialize the comparison processor for LLaVA
@st.cache_resource
def load_comparison_processor():
    """Load the thermal VLM comparison processor with caching"""
    return ThermalVLMComparison()

# Initialize the ensemble processor
@st.cache_resource
def load_ensemble_processor():
    """Load the thermal VLM ensemble processor with caching"""
    return ThermalVLMEnsemble()

@st.cache_resource
def load_smolvlm_processor():
    """Load SmolVLM processor with caching"""
    # Get Hugging Face token from multiple sources
    hf_token = None
    
    # Try to get from session state first
    hf_token = st.session_state.get("hf_token", None)
    
    # Try to get from environment variable
    if not hf_token:
        import os
        hf_token = os.getenv("HF_TOKEN", None)
    
    # Try to get from secrets
    if not hf_token:
        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
        except Exception:
            # Secrets not available, continue to next method
            pass
    
    # Try to get from .env file
    if not hf_token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN", None)
        except ImportError:
            pass
    
    # Try to get from token file
    if not hf_token:
        try:
            with open("hf_token.txt", "r") as f:
                hf_token = f.read().strip()
        except FileNotFoundError:
            pass
    
    return SmolVLMProcessor(hf_token=hf_token)

@st.cache_resource
def load_video_processor():
    """Load video processor with caching"""
    # Get Hugging Face token from multiple sources
    hf_token = None
    
    # Try to get from session state first
    hf_token = st.session_state.get("hf_token", None)
    
    # Try to get from environment variable
    if not hf_token:
        import os
        hf_token = os.getenv("HF_TOKEN", None)
    
    # Try to get from secrets
    if not hf_token:
        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
        except Exception:
            # Secrets not available, continue to next method
            pass
    
    # Try to get from .env file
    if not hf_token:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN", None)
        except ImportError:
            pass
    
    # Try to get from token file
    if not hf_token:
        try:
            with open("hf_token.txt", "r") as f:
                hf_token = f.read().strip()
        except FileNotFoundError:
            pass
    
    return ThermalVideoProcessor(hf_token=hf_token)

@st.cache_resource
def load_simple_video_processor():
    """Load simple video processor with caching"""
    return SimpleThermalVideoProcessor()

def main():
    # Enhanced header with better styling
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #ff6b35; font-size: 3rem; margin-bottom: 10px;">🔥 Thermal Image AI Analyzer</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">Advanced thermal image analysis powered by Vision-Language Models (VLM)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar header
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0; text-align: center;">⚙️ Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Hugging Face Token input
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 8px; border-radius: 8px; margin: 15px 0 10px 0;">
        <h4 style="color: white; margin: 0; text-align: center;">🔑 Hugging Face Token</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if token is already available
    token_available = False
    token_source = ""
    
    # Check multiple sources
    if st.session_state.get("hf_token"):
        token_available = True
        token_source = "Session State"
    elif os.getenv("HF_TOKEN"):
        token_available = True
        token_source = "Environment Variable"
    else:
        try:
            if st.secrets.get("HF_TOKEN"):
                token_available = True
                token_source = "Secrets"
        except Exception:
            pass
        
        # Check token file if still not available
        if not token_available:
            try:
                with open("hf_token.txt", "r") as f:
                    if f.read().strip():
                        token_available = True
                        token_source = "Token File"
            except FileNotFoundError:
                pass
    
    if token_available:
        st.sidebar.success(f"🔑 Token loaded from: {token_source}")
        st.sidebar.info("💡 Token is automatically available - no need to enter it again!")
        
        # Still allow manual override
        if st.sidebar.checkbox("Override token (optional)"):
            hf_token = st.sidebar.text_input(
                "Override Hugging Face Token:",
                type="password",
                help="Enter a different token if needed",
                placeholder="hf_..."
            )
            if hf_token:
                st.session_state.hf_token = hf_token
                st.sidebar.success("✅ Token overridden!")
    else:
        # Token input with help text
        hf_token = st.sidebar.text_input(
            "Hugging Face Token:",
            type="password",
            help="Enter your Hugging Face token to access models. Get one at https://huggingface.co/settings/tokens",
            placeholder="hf_..."
        )
        
        if hf_token:
            st.session_state.hf_token = hf_token
            st.sidebar.success("✅ Token saved!")
    
    # Model selection - now including LLaVA, SmolVLM and Ensemble
    model_options = {
        "BLIP Base": "Salesforce/blip-image-captioning-base",
        "BLIP Large": "Salesforce/blip-image-captioning-large",
        "GIT Base": "microsoft/git-base",
        "LLaVA-Next": "llava-next",  # New LLaVA option
        "SmolVLM": "lightweight_vlm",  # SmolVLM with advanced VLM approaches
        "Ensemble (All Models)": "Ensemble"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose VLM Model:",
        list(model_options.keys()),
        index=0
    )
    
    # Ensemble method selection (only show when ensemble is selected)
    ensemble_method = "weighted_average"
    if selected_model == "Ensemble (All Models)":
        ensemble_method = st.sidebar.selectbox(
            "Ensemble Method:",
            ["weighted_average", "majority_vote", "best_model"],
            help="Choose how to combine multiple model outputs"
        )
    
    # Enhanced domain knowledge section
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #4ecdc4, #44a08d); padding: 8px; border-radius: 8px; margin: 20px 0 10px 0;">
        <h4 style="color: white; margin: 0; text-align: center;">🔬 Domain Knowledge</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize knowledge injector
    knowledge_injector = ThermalKnowledgeInjector()
    
    # Knowledge injection options
    knowledge_options = st.sidebar.multiselect(
        "Select domain knowledge:",
        knowledge_injector.get_available_knowledge(),
        default=["Expert Thermal Analysis", "Human Detection Expert"],
        help="Select domain-specific knowledge to enhance AI analysis"
    )
    
    # Show knowledge descriptions
    if knowledge_options:
        st.sidebar.markdown("**Selected Knowledge:**")
        for option in knowledge_options:
            desc = knowledge_injector.get_knowledge_description(option)
            st.sidebar.markdown(f"• **{option}**: {desc}")
    
    # Custom prompt
    custom_prompt = st.sidebar.text_area(
        "Custom Analysis Prompt (optional):",
        value="Analyze this thermal image. Describe what you see, including temperature patterns, objects, and any anomalies.",
        height=100
    )
    
    # Create enhanced prompt with knowledge injection
    enhanced_prompt = knowledge_injector.create_expert_prompt(custom_prompt, knowledge_options)
    quick_prompt = knowledge_injector.get_quick_analysis_prompt(knowledge_options)
    
    # Prompt selection
    prompt_type = st.sidebar.radio(
        "Prompt Type:",
        ["Quick Analysis", "Detailed Expert Analysis"],
        help="Choose between quick focused analysis or detailed expert analysis"
    )
    
    # Use appropriate prompt based on selection
    final_prompt = quick_prompt if prompt_type == "Quick Analysis" else enhanced_prompt
    
    # Show enhanced prompt if knowledge is injected
    if knowledge_options and final_prompt != custom_prompt:
        with st.sidebar.expander("🔍 View Enhanced Prompt"):
            st.markdown(f"**{prompt_type} prompt with domain knowledge:**")
            st.text_area(
                "Enhanced Prompt:",
                value=final_prompt,
                height=200,
                disabled=True,
                help="This is the enhanced prompt that will be sent to the AI models"
            )
    
    # Input type selection
    st.sidebar.markdown("""
    <div style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 8px; border-radius: 8px; margin: 15px 0 10px 0;">
        <h4 style="color: white; margin: 0; text-align: center;">📁 Input Type</h4>
    </div>
    """, unsafe_allow_html=True)
    
    input_type = st.sidebar.radio(
        "Select Input Type:",
        ["Image", "Video"],
        help="Choose between image or video analysis"
    )
    
    # Show current mode indicator
    if input_type == "Image":
        st.sidebar.success("📸 Image Analysis Mode Active")
    else:
        st.sidebar.success("🎬 Video Analysis Mode Active")
    
    # Enhanced main content header
    if input_type == "Image":
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 15px; border-radius: 15px; margin: 30px 0 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">📤 Upload Thermal Image</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader for images
        uploaded_file = st.file_uploader(
            "Choose a thermal image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a thermal image to analyze"
        )
        
        # Store video settings as None for image mode
        analysis_mode = None
        max_frames = None
        
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); padding: 15px; border-radius: 15px; margin: 30px 0 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">🎬 Upload Thermal Video</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Video analysis settings
        col1, col2 = st.columns(2)
        with col1:
            analysis_mode = st.selectbox(
                "Analysis Mode:",
                ["summary", "key_frames", "comprehensive"],
                index=0,  # Default to summary mode
                help="Summary: Quick overview (recommended), Key Frames: Analyze scene changes, Comprehensive: Detailed analysis"
            )
        with col2:
            max_frames = st.slider(
                "Max Frames to Analyze:",
                min_value=3,
                max_value=20,
                value=5,  # Default to 5 frames for faster processing
                help="Maximum number of frames to analyze (higher = more detailed but slower)"
            )
        
        # File uploader for videos
        uploaded_file = st.file_uploader(
            "Choose a thermal video file",
            type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v"],
            help="Upload a thermal video to analyze"
        )
        
        # Add helpful tips for video upload
        if not uploaded_file:
            st.info("💡 **Video Upload Tips:**")
            st.markdown("""
            - **Recommended format:** MP4 with H.264 codec
            - **File size:** Under 100MB for faster processing
            - **Duration:** Shorter videos (under 30 seconds) process faster
            - **Resolution:** 720p or lower for optimal performance
            - **First time?** Try the test video in the section below
            """)
    
    # Enhanced test files section
    if input_type == "Image":
        st.markdown("""
        <div style="background: linear-gradient(90deg, #f093fb, #f5576c); padding: 12px; border-radius: 12px; margin: 25px 0 15px 0;">
            <h3 style="color: white; margin: 0; text-align: center;">📁 Or Select from Test Images</h3>
        </div>
        """, unsafe_allow_html=True)
        test_folder = "test_image"
        if os.path.exists(test_folder):
            test_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                test_files.extend(Path(test_folder).glob(f"*{ext}"))
                test_files.extend(Path(test_folder).glob(f"*{ext.upper()}"))
            
            if test_files:
                selected_test_file = st.selectbox(
                    "Choose from test images:",
                    [img.name for img in test_files],
                    index=0 if test_files else None
                )
                
                if selected_test_file:
                    test_file_path = str(Path(test_folder) / selected_test_file)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔍 Analyze Selected Test Image", use_container_width=True):
                            analyze_image(test_file_path, final_prompt, selected_model, ensemble_method, model_options)
            else:
                st.markdown("""
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
                    <p style="margin: 0; color: #856404;">📁 No test images found in the test_image folder</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="margin: 0; color: #721c24;">⚠️ Test image folder not found</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #f093fb, #f5576c); padding: 12px; border-radius: 12px; margin: 25px 0 15px 0;">
            <h3 style="color: white; margin: 0; text-align: center;">📁 Or Select from Test Videos</h3>
        </div>
        """, unsafe_allow_html=True)
        test_folder = "test_video"
        if os.path.exists(test_folder):
            test_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']:
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
                            analyze_video(test_file_path, final_prompt, analysis_mode, max_frames)
            else:
                st.markdown("""
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0;">
                    <p style="margin: 0; color: #856404;">📁 No test videos found in the test_video folder</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="margin: 0; color: #721c24;">⚠️ Test video folder not found</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Handle uploaded file
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Analyze based on input type
        if input_type == "Image":
            analyze_image(temp_path, final_prompt, selected_model, ensemble_method, model_options)
        else:
            analyze_video(temp_path, final_prompt, analysis_mode, max_frames)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_image(image_path, custom_prompt, selected_model, ensemble_method="weighted_average", model_options=None):
    """Analyze a thermal image and display results"""
    
    try:
        if selected_model == "Ensemble (All Models)":
            # Use Ensemble analysis
            with st.spinner("Loading Ensemble models..."):
                processor = load_ensemble_processor()
            
            with st.spinner("Analyzing thermal image with Ensemble..."):
                result = processor.analyze_with_ensemble(image_path, custom_prompt, ensemble_method)
            
            if result:
                display_ensemble_results(result)
            else:
                st.error("Failed to analyze the image with Ensemble. Please try again.")
        elif selected_model == "LLaVA-Next":
            # Use LLaVA-Next analysis
            with st.spinner("Loading LLaVA-Next model..."):
                processor = load_comparison_processor()
            
            with st.spinner("Analyzing thermal image with LLaVA-Next..."):
                result = processor.compare_models(image_path, custom_prompt)
            
            if result:
                display_llava_results(result)
            else:
                st.error("Failed to analyze the image with LLaVA-Next. Please try again.")
        elif selected_model == "SmolVLM":
            # Use SmolVLM analysis
            with st.spinner("Loading SmolVLM model..."):
                processor = load_smolvlm_processor()
            
            with st.spinner("Analyzing thermal image with SmolVLM..."):
                result = processor.analyze_thermal_image(image_path, custom_prompt)
            
            if result:
                display_smolvlm_results(result)
            else:
                st.error("Failed to analyze the image with SmolVLM. Please try again.")
        else:
            # Use existing BLIP/GIT analysis
            with st.spinner(f"Loading {selected_model}..."):
                processor = ThermalImageProcessor()
                processor.load_model(model_options[selected_model])
            
            with st.spinner("Analyzing thermal image..."):
                result = processor.analyze_thermal_image(image_path, custom_prompt)
            
            if result:
                display_standard_results(result)
            else:
                st.error("Failed to analyze the image. Please try again.")
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error during analysis: {error_msg}")
        
        # Provide specific help based on error type
        if "No secrets files found" in error_msg:
            st.info("💡 This is a configuration issue, not a dependency problem. The app will work with the token file.")
        elif "No module named" in error_msg:
            st.info("Make sure you have the required dependencies installed: pip install -r requirements.txt")
        else:
            st.info("💡 Try using a different model or check the image format.")

def analyze_video(video_path, custom_prompt, analysis_mode="summary", max_frames=10):
    """Analyze a thermal video and display results"""
    
    try:
        # Start with simple processor for better reliability
        try:
            with st.spinner("Loading simplified video processor..."):
                simple_processor = load_simple_video_processor()
            
            # Update processor settings for faster processing
            simple_processor.max_frames_to_analyze = min(max_frames, 10)  # Limit for simple processor
            
            with st.spinner(f"Analyzing video with simplified {analysis_mode} mode..."):
                result = simple_processor.process_video(video_path, custom_prompt, analysis_mode)
            
            if result and 'error' not in result:
                display_video_results(result)
                st.success("✅ Analysis completed using simplified processor")
                return
            else:
                error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
                st.warning(f"Simplified analysis failed: {error_msg}")
                st.info("🔄 Trying advanced analysis...")
                
        except Exception as e:
            st.warning(f"Simplified analysis failed: {str(e)}")
            st.info("🔄 Trying advanced analysis...")
        
        # Fallback to advanced processor only if simple fails
        try:
            with st.spinner("Loading advanced video processor..."):
                processor = load_video_processor()
            
            # Update processor settings
            processor.max_frames_to_analyze = max_frames
            
            with st.spinner(f"Analyzing thermal video with {analysis_mode} mode..."):
                result = processor.process_video(video_path, custom_prompt, analysis_mode)
            
            if result and 'error' not in result:
                display_video_results(result)
                st.success("✅ Analysis completed using advanced processor")
            else:
                error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
                st.error(f"Advanced analysis also failed: {error_msg}")
        
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error during advanced video analysis: {error_msg}")
            st.info("💡 Make sure the video file is valid and in a supported format.")
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error during video analysis: {error_msg}")
        st.info("💡 Make sure the video file is valid and in a supported format.")

def display_video_results(result):
    """Display video analysis results"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); padding: 15px; border-radius: 15px; margin: 20px 0;">
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
    
    # Video summary
    st.subheader("📋 Video Summary")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #ff6b35;
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
    
    # Temporal analysis
    temporal_analysis = result['temporal_analysis']
    if 'error' not in temporal_analysis:
        st.subheader("📈 Temporal Analysis")
        
        # Temperature trend
        if 'temperature_trend' in temporal_analysis:
            trend = temporal_analysis['temperature_trend']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature Trend", trend['trend'].title())
            with col2:
                st.metric("Temperature Change", f"{trend.get('temperature_change', 0):.2f}")
            with col3:
                st.metric("Correlation", f"{trend.get('correlation', 0):.3f}")
        
        # Motion analysis
        if 'motion_patterns' in temporal_analysis:
            motion = temporal_analysis['motion_patterns']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Motion Detected", "Yes" if motion['motion_detected'] else "No")
            with col2:
                st.metric("Motion Percentage", f"{motion['motion_percentage']:.1f}%")
        
        # Content consistency
        if 'content_consistency' in temporal_analysis:
            consistency = temporal_analysis['content_consistency']
            st.metric("Content Consistency", f"{consistency['consistency_score']:.2f}")
            
            if consistency['common_terms']:
                st.markdown("**Common Elements:**")
                st.write(", ".join(consistency['common_terms'][:10]))
    
    # Frame-by-frame analysis
    if result['frame_analyses']:
        st.subheader("🎞️ Frame-by-Frame Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "🎯 Key Frames", "📈 Temperature Graph"])
        
        with tab1:
            # Summary table
            summary_data = []
            for analysis in result['frame_analyses']:
                summary_data.append({
                    "Frame": analysis['frame_number'],
                    "Time (s)": f"{analysis['timestamp']:.2f}",
                    "Mean Temp": f"{analysis['temperature_analysis']['mean_temperature']:.1f}",
                    "Description": analysis['caption'][:100] + "..." if len(analysis['caption']) > 100 else analysis['caption']
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            # Show key frames with descriptions
            for i, analysis in enumerate(result['frame_analyses'][:10]):  # Show first 10 frames
                with st.expander(f"Frame {analysis['frame_number']} (t={analysis['timestamp']:.2f}s)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Temperature:** {analysis['temperature_analysis']['mean_temperature']:.1f}")
                        st.markdown(f"**Processing Time:** {analysis.get('processing_time', 0):.2f}s")
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
        
        with tab3:
            # Temperature graph over time
            if len(result['frame_analyses']) > 1:
                timestamps = [analysis['timestamp'] for analysis in result['frame_analyses']]
                temperatures = [analysis['temperature_analysis']['mean_temperature'] for analysis in result['frame_analyses']]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(timestamps, temperatures, 'o-', color='#ff6b35', linewidth=2, markersize=6)
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Mean Temperature')
                ax.set_title('Temperature Variation Over Time')
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8f9fa')
                
                # Add trend line if available
                if 'temperature_trend' in temporal_analysis and temporal_analysis['temperature_trend']['trend'] != 'insufficient_data':
                    trend = temporal_analysis['temperature_trend']
                    if len(timestamps) > 1:
                        z = np.polyfit(timestamps, temperatures, 1)
                        p = np.poly1d(z)
                        ax.plot(timestamps, p(timestamps), "--", color='#f7931e', alpha=0.7, label=f"Trend: {trend['trend']}")
                        ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Anomalies
    if 'temporal_anomalies' in temporal_analysis and temporal_analysis['temporal_anomalies']:
        st.subheader("⚠️ Detected Anomalies")
        anomalies = temporal_analysis['temporal_anomalies']
        
        for anomaly in anomalies[:5]:  # Show top 5 anomalies
            with st.expander(f"{anomaly['type'].replace('_', ' ').title()} at {anomaly['timestamp']:.2f}s"):
                st.markdown(f"**Frame:** {anomaly['frame_number']}")
                st.markdown(f"**Description:** {anomaly['description']}")
                if 'severity' in anomaly:
                    st.markdown(f"**Severity:** {anomaly['severity']:.2f}")
    
    # Processing information
    st.subheader("⚙️ Processing Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analysis Mode", result['analysis_mode'].title())
    with col2:
        st.metric("Total Processing Time", f"{result['processing_time']:.2f}s")
    with col3:
        st.metric("Frames per Second", f"{result['total_frames_analyzed']/result['processing_time']:.1f}")

def display_ensemble_results(result):
    """Display ensemble analysis results"""
    st.header("🎯 Ensemble Analysis Results")
    
    # Display the processed image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Processed Thermal Image")
        if 'enhanced_image' in result:
            st.image(result['enhanced_image'], caption="Edge-enhanced thermal image", use_column_width=True)
        elif 'image' in result and result['image']:
            st.image(result['image'], caption="Edge-enhanced thermal image", use_column_width=True)
        else:
            st.warning("No image data available for display")
    
    with col2:
        st.subheader("🌡️ Temperature Analysis")
        temp_analysis = result['temperature_analysis']
        st.metric("Mean Temp", f"{temp_analysis['mean_temperature']:.1f}")
        st.metric("Max Temp", f"{temp_analysis['max_temperature']:.1f}")
        st.metric("Min Temp", f"{temp_analysis['min_temperature']:.1f}")
    
    # Display ensemble caption
    st.subheader("🎯 Ensemble AI Description")
    
    # Use a styled container for better visibility
    st.markdown("### Full Ensemble Analysis:")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #00ff88;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        ">
        {result['ensemble_caption']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display ensemble method and processing info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Ensemble Method:** {result['ensemble_method'].replace('_', ' ').title()}")
    with col2:
        st.info(f"**Total Processing Time:** {result['total_processing_time']:.2f}s")
    
    # Display individual model results
    st.subheader("🔍 Individual Model Results")
    
    # Create tabs for each model
    tab_names = list(result['individual_results'].keys())
    tabs = st.tabs(tab_names)
    
    for i, (model_name, model_result) in enumerate(result['individual_results'].items()):
        with tabs[i]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Model:** {model_name}")
                st.markdown(f"**Confidence:** {model_result['confidence']:.3f}")
                st.markdown(f"**Processing Time:** {model_result['processing_time']:.2f}s")
                
                # Show confidence breakdown if available
                if 'confidence_breakdown' in model_result and model_result['confidence_breakdown']:
                    st.markdown("**Confidence Breakdown:**")
                    breakdown = model_result['confidence_breakdown']
                    for factor, score in breakdown.items():
                        if factor != 'Overall Confidence':
                            st.markdown(f"• {factor}: {score}")
            
            with col2:
                st.markdown("**AI Description:**")
                # Use styled container for better visibility
                st.markdown(
                    f"""
                    <div style="
                        background-color: #2d2d2d; 
                        color: #e0e0e0; 
                        padding: 15px; 
                        border-radius: 8px; 
                        border-left: 4px solid #4CAF50;
                        font-family: 'Arial', sans-serif;
                        font-size: 13px;
                        line-height: 1.5;
                        white-space: pre-wrap;
                        max-height: 200px;
                        overflow-y: auto;
                    ">
                    {model_result['caption']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Display model weights
    st.subheader("⚖️ Model Weights")
    weights_df = pd.DataFrame([
        {"Model": model, "Weight": f"{weight:.2%}"} 
        for model, weight in result['model_weights'].items()
    ])
    st.dataframe(weights_df, use_container_width=True)
    
    # Display insights
    st.subheader("💡 Ensemble Insights")
    insights = []
    
    # Confidence analysis
    confidences = [result['individual_results'][model]['confidence'] for model in result['individual_results']]
    avg_confidence = sum(confidences) / len(confidences)
    insights.append(f"**Average Confidence:** {avg_confidence:.2f}")
    
    # Best performing model
    best_model = max(result['individual_results'].items(), key=lambda x: x[1]['confidence'])
    insights.append(f"**Best Individual Model:** {best_model[0]} (Confidence: {best_model[1]['confidence']:.2f})")
    
    # Processing time analysis
    total_time = result['total_processing_time']
    insights.append(f"**Total Processing Time:** {total_time:.2f}s")
    
    # Ensemble method effectiveness
    if result['ensemble_method'] == 'weighted_average':
        insights.append("**Ensemble Method:** Weighted average provides balanced analysis")
    elif result['ensemble_method'] == 'majority_vote':
        insights.append("**Ensemble Method:** Majority vote identifies common themes")
    else:
        insights.append("**Ensemble Method:** Best model selection for optimal performance")
    
    for insight in insights:
        st.markdown(f"• {insight}")

def display_standard_results(result):
    """Display results for BLIP/GIT models"""
    st.header("📊 Analysis Results")
    
    # Display original image
    st.subheader("Original Image")
    if 'enhanced_image' in result:
        st.image(result['enhanced_image'], caption="Processed Thermal Image", use_column_width=True)
    elif 'image' in result:
        st.image(result['image'], caption="Processed Thermal Image", use_column_width=True)
    else:
        st.warning("No image data available for display")
    
        # Display caption
    st.subheader("🔍 AI Description")
    st.markdown(
        f"""
        <div style="
            background-color: #1e3a8a; 
            color: #ffffff; 
            padding: 18px; 
            border-radius: 10px; 
            border-left: 5px solid #3b82f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
        {result['caption']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display temperature analysis
    st.subheader("🌡️ Temperature Analysis")
    temp_data = result['temperature_analysis']
    
    # Create metrics in a row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Temp", f"{temp_data['mean_temperature']:.1f}")
    with col2:
        st.metric("Max Temp", f"{temp_data['max_temperature']:.1f}")
    with col3:
        st.metric("Min Temp", f"{temp_data['min_temperature']:.1f}")
    
    # Temperature distribution
    st.subheader("📈 Temperature Distribution")
    temp_stats = {
        "Hot Regions": temp_data['hot_regions_percentage'],
        "Cold Regions": temp_data['cold_regions_percentage'],
        "Temperature Range": temp_data['temperature_range']
    }
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(temp_stats.keys())
    values = list(temp_stats.values())
    bars = ax.bar(keys, values, color=['red', 'blue', 'orange'])
    ax.set_ylabel('Percentage / Value')
    ax.set_title('Temperature Analysis')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed statistics
    st.subheader("📊 Detailed Statistics")
    for key, value in temp_data.items():
        st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")

def display_llava_results(result):
    """Display results for LLaVA-Next model with comparison"""
    st.header("📊 LLaVA-Next Analysis Results")
    
    # Display the thermal image
    st.subheader("📸 Thermal Image")
    if 'enhanced_image' in result:
        st.image(result['enhanced_image'], caption="Processed Thermal Image", use_column_width=True)
    elif 'image' in result:
        st.image(result['image'], caption="Processed Thermal Image", use_column_width=True)
    else:
        st.warning("No image data available for display")
    
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
        st.markdown(f"**VLM Used:** {'✅ Yes' if result['blip_result']['vlm_used'] else '❌ No (Fallback)'}")
        
        # Display BLIP caption
        st.markdown("**AI Description:**")
        st.markdown(
            f"""
            <div style="
                background-color: #dc2626; 
                color: #ffffff; 
                padding: 15px; 
                border-radius: 8px; 
                border-left: 4px solid #ef4444;
                font-family: 'Arial', sans-serif;
                font-size: 13px;
                line-height: 1.5;
                white-space: pre-wrap;
                max-height: 120px;
                overflow-y: auto;
            ">
            {result['blip_result']['caption']}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("### 🚀 LLaVA-Next Analysis")
        st.markdown(f"**Model:** {result['llava_result']['model']}")
        st.markdown(f"**Processing Time:** {result['llava_result']['processing_time']:.2f}s")
        st.markdown(f"**VLM Used:** {'✅ Yes' if result['llava_result']['vlm_used'] else '❌ No (Fallback)'}")
        
        # Display LLaVA-Next caption
        st.markdown("**AI Description:**")
        st.markdown(
            f"""
            <div style="
                background-color: #059669; 
                color: #ffffff; 
                padding: 15px; 
                border-radius: 8px; 
                border-left: 4px solid #10b981;
                font-family: 'Arial', sans-serif;
                font-size: 13px;
                line-height: 1.5;
                white-space: pre-wrap;
                max-height: 120px;
                overflow-y: auto;
            ">
            {result['llava_result']['caption']}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Comparison metrics
    st.subheader("📊 Comparison Metrics")
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Processing time comparison
    models = ['BLIP', 'LLaVA-Next']
    times = [result['blip_result']['processing_time'], result['llava_result']['processing_time']]
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
    lengths = [len(result['blip_result']['caption']), len(result['llava_result']['caption'])]
    
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
    
    # Analysis insights
    st.subheader("💡 Analysis Insights")
    
    # Determine which model performed better
    blip_time = result['blip_result']['processing_time']
    llava_time = result['llava_result']['processing_time']
    blip_length = len(result['blip_result']['caption'])
    llava_length = len(result['llava_result']['caption'])
    
    insights = []
    
    if blip_time < llava_time:
        insights.append("🔥 **BLIP is faster** - Better for real-time applications")
    else:
        insights.append("🚀 **LLaVA-Next is faster** - More efficient processing")
    
    if blip_length < llava_length:
        insights.append("🚀 **LLaVA-Next provides more detailed descriptions** - Better for comprehensive analysis")
    else:
        insights.append("🔥 **BLIP provides concise descriptions** - Good for quick overview")
    
    # Check VLM usage
    if result['blip_result']['vlm_used'] and result['llava_result']['vlm_used']:
        insights.append("✅ **Both models successfully used VLM** - Good thermal image compatibility")
    elif result['blip_result']['vlm_used']:
        insights.append("⚠️ **BLIP used VLM, LLaVA-Next used fallback** - BLIP better for this image")
    elif result['llava_result']['vlm_used']:
        insights.append("⚠️ **LLaVA-Next used VLM, BLIP used fallback** - LLaVA-Next better for this image")
    else:
        insights.append("❌ **Both models used fallback** - Thermal images challenging for VLMs")
    
    # Check for repetitive patterns
    if "thermal thermal" in result['blip_result']['caption'].lower():
        insights.append("⚠️ **BLIP shows repetitive patterns** - Common issue with thermal images")
    
    if "struggled" in result['blip_result']['caption'].lower():
        insights.append("❌ **BLIP struggled with this thermal image** - LLaVA-Next may be more robust")
    
    for insight in insights:
        st.markdown(insight)

def display_smolvlm_results(result):
    """Display SmolVLM analysis results"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4ecdc4, #44a08d); padding: 15px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🔬 SmolVLM Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the processed image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Processed Thermal Image")
        if result['enhanced_image'] is not None:
            # Convert numpy array to PIL Image for display
            import numpy as np
            from PIL import Image
            enhanced_img = Image.fromarray((result['enhanced_image'] * 255).astype(np.uint8))
            st.image(enhanced_img, caption="SmolVLM processed image", use_column_width=True)
    
    with col2:
        st.subheader("🌡️ Temperature Analysis")
        temp_analysis = result['temperature_analysis']
        st.metric("Mean Temp", f"{temp_analysis['mean_temperature']:.1f}")
        st.metric("Max Temp", f"{temp_analysis['max_temperature']:.1f}")
        st.metric("Min Temp", f"{temp_analysis['min_temperature']:.1f}")
    
    # Display SmolVLM caption
    st.subheader("🔬 SmolVLM AI Description")
    
    # Use a styled container for better visibility
    st.markdown("### Full SmolVLM Analysis:")
    st.markdown(
        f"""
        <div style="
            background-color: #1f1f1f; 
            color: #ffffff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #4ecdc4;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        ">
        {result['caption']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Enhanced processing info display
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 10px; border-radius: 10px; margin: 20px 0;">
        <h4 style="color: white; margin: 0; text-align: center;">📊 Processing Information</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; color: #1976d2; font-weight: bold;">🤖 Model: {result['model']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; color: #388e3c; font-weight: bold;">⏱️ Processing Time: {result['processing_time']:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced temperature statistics header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); padding: 12px; border-radius: 12px; margin: 25px 0 15px 0;">
        <h3 style="color: white; margin: 0; text-align: center;">📊 Detailed Temperature Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    temp_analysis = result['temperature_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Temperature", f"{temp_analysis['mean_temperature']:.2f}")
    with col2:
        st.metric("Temperature Std", f"{temp_analysis['temperature_std']:.2f}")
    with col3:
        st.metric("Min Temperature", f"{temp_analysis['min_temperature']:.2f}")
    with col4:
        st.metric("Max Temperature", f"{temp_analysis['max_temperature']:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hot Regions %", f"{temp_analysis['hot_regions_percentage']:.2f}%")
    with col2:
        st.metric("Cold Regions %", f"{temp_analysis['cold_regions_percentage']:.2f}%")
    with col3:
        st.metric("Temperature Range", f"{temp_analysis['temperature_range']:.2f}")
    with col4:
        st.metric("Human Patterns", f"{temp_analysis['potential_human_patterns']}")
    
    # Enhanced advanced analysis header
    if 'thermal_gradients' in temp_analysis and 'thermal_anomalies_percentage' in temp_analysis:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #f093fb, #f5576c); padding: 12px; border-radius: 12px; margin: 25px 0 15px 0;">
            <h3 style="color: white; margin: 0; text-align: center;">🔬 Advanced Thermal Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Thermal Gradients", f"{temp_analysis['thermal_gradients']:.2f}")
        with col2:
            st.metric("Thermal Anomalies %", f"{temp_analysis['thermal_anomalies_percentage']:.2f}%")

    # Add a footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;">
        <p style="margin: 0; color: #666; font-size: 0.9rem;">
            🔥 <strong>Thermal Image AI Analyzer</strong> | Powered by Vision-Language Models | 
            <a href="#" style="color: #ff6b35; text-decoration: none;">Documentation</a> | 
            <a href="#" style="color: #ff6b35; text-decoration: none;">Support</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
