# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### ❌ "ModuleNotFoundError: No module named 'seaborn'"

**Problem:** The app fails to start because seaborn is missing.

**Solutions:**
1. **Install seaborn in conda environment:**
   ```bash
   conda install seaborn -y
   ```

2. **Or install via pip:**
   ```bash
   pip install seaborn
   ```

3. **The app now handles missing seaborn gracefully** - it will work without it.

### ❌ "Bad message format" / "Tried to use SessionInfo before it was initialized"

**Problem:** Streamlit session initialization error.

**Solutions:**
1. **Restart the Streamlit app:**
   ```bash
   # Stop the current app (Ctrl+C)
   # Then restart:
   streamlit run streamlit_app.py
   ```

2. **Clear Streamlit cache:**
   ```bash
   streamlit cache clear
   ```

3. **Check for multiple Streamlit instances:**
   ```bash
   # On Windows:
   tasklist | findstr streamlit
   # Kill any existing processes
   ```

### ❌ "video/mp4 files are not allowed"

**Problem:** Trying to upload video in Image mode.

**Solution:**
1. **Switch to Video mode** in the sidebar
2. **Look for "🎬 Video Analysis Mode Active"** in green
3. **Upload your video file**

### ❌ Video Analysis Stops/Hangs

**Problem:** Analysis gets stuck during processing.

**Solutions:**
1. **Use "summary" mode** instead of "comprehensive"
2. **Reduce frame count** to 5-10 frames
3. **Try shorter videos** (under 30 seconds)
4. **Close other applications** to free up memory

### ❌ "No module named 'thermal_video_processor'"

**Problem:** Missing custom modules.

**Solution:**
1. **Ensure all files are in the same directory:**
   - `streamlit_app.py`
   - `thermal_video_processor.py`
   - `thermal_video_processor_simple.py`
   - `thermal_vlm_processor.py`
   - etc.

2. **Check file permissions** - ensure files are readable

### ❌ Hugging Face Token Issues

**Problem:** Models fail to load due to token issues.

**Solutions:**
1. **Add your token to `hf_token.txt`:**
   ```
   hf_your_token_here
   ```

2. **Or set environment variable:**
   ```bash
   set HF_TOKEN=hf_your_token_here
   ```

3. **Or enter token in the app sidebar**

## Quick Fix Commands

### 🔧 Install All Dependencies
```bash
python setup_dependencies.py
```

### 🧪 Test Basic Setup
```bash
python test_app_simple.py
```

### 🎬 Test Video Processing
```bash
python test_simple_video.py
```

### 🔍 Debug Video Upload
```bash
python debug_video_upload.py
```

## Performance Optimization

### 🚀 Faster Processing
- Use "summary" analysis mode
- Set frame count to 5-10
- Use shorter videos (< 30 seconds)
- Lower resolution videos (720p or less)

### 💾 Memory Management
- Close other applications
- Process one video at a time
- Restart app if memory usage is high

## Supported Video Formats

| Format | Extension | Status |
|--------|-----------|--------|
| MP4    | .mp4      | ✅ Full support |
| AVI    | .avi      | ✅ Full support |
| MOV    | .mov      | ✅ Full support |
| MKV    | .mkv      | ⚠️ Limited |
| WMV    | .wmv      | ⚠️ Limited |
| FLV    | .flv      | ⚠️ Limited |
| WebM   | .webm     | ⚠️ Limited |
| M4V    | .m4v      | ✅ Full support |

## System Requirements

### Minimum Requirements
- **RAM:** 4GB available
- **Storage:** 2GB free space
- **Python:** 3.8 or higher
- **OS:** Windows 10/11, macOS, Linux

### Recommended Requirements
- **RAM:** 8GB or more
- **Storage:** 5GB free space
- **GPU:** CUDA-compatible (optional, for faster processing)

## Getting Help

### 📋 Before Asking for Help
1. **Run the test script:** `python test_app_simple.py`
2. **Check the debug output:** `python debug_video_upload.py`
3. **Try with a test video** first
4. **Check system resources** (RAM, disk space)

### 🆘 Still Having Issues?
1. **Check the error messages** carefully
2. **Try the simplified processor** first
3. **Use smaller test files**
4. **Restart the application**

---

**💡 Pro Tip:** Always start with the test scripts to verify your setup is working correctly! 