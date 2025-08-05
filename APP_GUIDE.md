# 🔥 Thermal Image & Escalator Safety Analysis Suite

## 📱 Available Applications

### 1. 🔥 **Thermal Image Analyzer** (Original)
- **File**: `streamlit_app.py`
- **Port**: 8501
- **Purpose**: Original thermal image and video analysis
- **Features**: 
  - Thermal image processing
  - Temperature analysis
  - VLM model integration (BLIP, GIT, LLaVA-Next, SmolVLM)
  - Video analysis with multiple modes

### 2. 🤖 **VLM-Enhanced Escalator Safety Monitor**
- **File**: `streamlit_escalator_vlm.py`
- **Port**: 8502
- **Purpose**: AI-powered escalator safety using Vision-Language Models
- **Features**:
  - VLM-enhanced crowding detection
  - AI-powered falling object detection
  - Real-time safety alerts
  - Frame-by-frame AI analysis
  - Customizable safety thresholds

### 3. 🚇 **Basic Escalator Safety Monitor**
- **File**: `streamlit_escalator_safety.py`
- **Port**: 8503
- **Purpose**: Fast escalator safety monitoring without heavy AI models
- **Features**:
  - Basic computer vision for crowding detection
  - Blob detection for falling objects
  - Fast processing
  - Safety alerts and metrics

### 4. 🎬 **General Video Analyzer**
- **File**: `streamlit_general_video.py`
- **Port**: 8504
- **Purpose**: General video analysis (non-thermal)
- **Features**:
  - Motion detection
  - Scene change analysis
  - Brightness and contrast analysis
  - Frame-by-frame statistics

### 5. ⚡ **Fast Thermal Video Processor**
- **File**: `streamlit_app_fast.py`
- **Port**: 8505
- **Purpose**: Optimized thermal video processing
- **Features**:
  - Fast thermal analysis
  - Simplified processing pipeline
  - Reduced frame count for speed
  - Basic thermal metrics

## 🚀 How to Run

### Option 1: Easy Launcher (Recommended)
```bash
python launch_apps.py
```
This will show you a menu to choose which app to launch.

### Option 2: Direct Launch
```bash
# Thermal Image Analyzer
streamlit run streamlit_app.py --server.port 8501

# VLM-Enhanced Escalator Safety
streamlit run streamlit_escalator_vlm.py --server.port 8502

# Basic Escalator Safety
streamlit run streamlit_escalator_safety.py --server.port 8503

# General Video Analyzer
streamlit run streamlit_general_video.py --server.port 8504

# Fast Thermal Video
streamlit run streamlit_app_fast.py --server.port 8505
```

## 🎯 Recommended Apps for Your Use Case

### For Thermal Images:
- **Use**: `streamlit_app.py` (Port 8501)
- **Best for**: Professional thermal analysis with full VLM capabilities

### For Escalator Safety (Your Main Request):
- **Use**: `streamlit_escalator_vlm.py` (Port 8502)
- **Best for**: AI-enhanced detection of crowding and falling objects
- **Features**: Uses Vision-Language Models for better accuracy

## 🔧 Configuration

### Hugging Face Token (for VLM apps)
To enable AI-enhanced analysis, you need a Hugging Face token:

1. **Get token**: Visit https://huggingface.co/settings/tokens
2. **Add to app**: Enter in the sidebar of VLM apps
3. **Or save to file**: Create `hf_token.txt` with your token

### Test Videos
Place your test videos in the `test_video/` folder for easy access.

## 📊 App Comparison

| App | Speed | AI Enhancement | Best For | Port |
|-----|-------|----------------|----------|------|
| Thermal Image | Medium | ✅ Full VLM | Thermal analysis | 8501 |
| VLM Escalator | Medium | ✅ VLM Enhanced | Escalator safety | 8502 |
| Basic Escalator | Fast | ❌ Basic CV | Quick safety check | 8503 |
| General Video | Fast | ❌ Basic CV | General video | 8504 |
| Fast Thermal | Fast | ❌ Basic CV | Quick thermal | 8505 |

## 🎯 Quick Start for Escalator Safety

1. **Run the launcher**:
   ```bash
   python launch_apps.py
   ```

2. **Choose option 2** (VLM-Enhanced Escalator Safety)

3. **Upload your escalator video** or select from test videos

4. **Configure settings** in the sidebar:
   - Add Hugging Face token for AI enhancement
   - Adjust frame count (8-12 recommended)
   - Set safety thresholds

5. **Analyze** and view results:
   - Safety alerts
   - Crowding detection
   - Falling object risk
   - AI insights

## 🚨 Safety Alerts

The escalator safety apps will show:
- **🚨 HIGH CROWDING**: Immediate attention required
- **⚠️ MODERATE CROWDING**: Monitor situation
- **🚨 HIGH FALLING RISK**: Objects/bags may fall
- **⚠️ MODERATE FALLING RISK**: Monitor for dropped items
- **✅ SAFE**: No safety alerts

## 💡 Tips

1. **For best VLM results**: Use 8-12 frames for analysis
2. **For speed**: Use basic escalator safety (Port 8503)
3. **For accuracy**: Use VLM-enhanced escalator safety (Port 8502)
4. **Test first**: Try with small video files before large ones
5. **Token required**: VLM apps need Hugging Face token for AI features

## 🔍 Troubleshooting

- **App won't start**: Check if port is already in use
- **VLM not working**: Ensure Hugging Face token is provided
- **Slow processing**: Reduce frame count or use basic apps
- **No test videos**: Place videos in `test_video/` folder

## 📞 Support

If you encounter issues:
1. Check the app logs in the terminal
2. Try the basic version first
3. Ensure all dependencies are installed
4. Verify video file format is supported 