# 🔥 Thermal Video Upload and Analysis Guide

## Quick Start

### 1. Start the Application
```bash
streamlit run streamlit_app.py
```

### 2. Select Video Input
- In the sidebar, change **"Input Type"** from "Image" to **"Video"**

### 3. Configure Analysis Settings
- **Analysis Mode:**
  - `summary` - Quick overview (recommended for first try)
  - `key_frames` - Analyze scene changes
  - `comprehensive` - Detailed analysis (slower)
- **Max Frames to Analyze:** 5-50 (higher = more detailed but slower)

### 4. Upload Your Video
- Click **"Browse files"** in the video upload section
- Select your thermal video file
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`

### 5. Wait for Analysis
- The system will automatically process your video
- You'll see progress indicators during processing
- If advanced analysis fails, it will automatically try simplified analysis

## Troubleshooting Common Issues

### ❌ Video Analysis Stops/Hangs

**Problem:** Analysis gets stuck at "Analyzing thermal video with comprehensive mode..."

**Solutions:**
1. **Try Simplified Mode:**
   - Change Analysis Mode to "summary"
   - Reduce Max Frames to 5-10
   
2. **Check Video File:**
   - Ensure video is not corrupted
   - Try converting to MP4 format
   - Check file size (should be < 100MB for faster processing)

3. **System Resources:**
   - Close other applications to free up memory
   - Ensure you have at least 2GB free RAM

### ❌ "Invalid video file" Error

**Solutions:**
1. **Convert Video Format:**
   ```bash
   # Using ffmpeg (if available)
   ffmpeg -i input.avi -c:v libx264 output.mp4
   ```

2. **Check File Permissions:**
   - Ensure the video file is readable
   - Try copying to a different location

3. **File Size Issues:**
   - Try a shorter video (under 30 seconds)
   - Reduce video resolution

### ❌ "No valid frames extracted" Error

**Solutions:**
1. **Video Codec Issues:**
   - Convert to H.264 codec
   - Use MP4 container format

2. **Corrupted Video:**
   - Try a different video file
   - Check if video plays in media player

## Analysis Results Explained

### 📹 Video Information
- **Duration:** Video length in seconds
- **Resolution:** Video dimensions
- **FPS:** Frames per second
- **Frames Analyzed:** Number of frames processed

### 📋 Video Summary
- Comprehensive text analysis of the entire video
- Temperature trends and patterns
- Motion detection and analysis
- Key insights and observations

### 📈 Temporal Analysis
- **Temperature Trend:** Increasing, decreasing, or stable
- **Motion Patterns:** Percentage of frames with motion
- **Content Consistency:** How similar frames are to each other

### 🎞️ Frame-by-Frame Analysis
- Individual frame descriptions
- Temperature data for each analyzed frame
- Interactive tabs for different views:
  - **Summary:** Table view of all frames
  - **Key Frames:** Detailed frame analysis
  - **Temperature Graph:** Visual temperature trends

### ⚠️ Anomaly Detection
- Temperature spikes and unusual patterns
- Content changes and scene transitions
- Severity ratings for detected anomalies

## Performance Tips

### 🚀 Faster Processing
1. **Use "summary" mode** for quick results
2. **Reduce frame count** to 5-10 frames
3. **Use shorter videos** (under 30 seconds)
4. **Lower resolution videos** (720p or less)

### 🎯 Better Results
1. **Use "comprehensive" mode** for detailed analysis
2. **Increase frame count** to 20-30 frames
3. **Use higher resolution** videos
4. **Ensure good video quality**

### 💾 Memory Management
1. **Close other applications** during analysis
2. **Use shorter videos** if you have limited RAM
3. **Process videos one at a time**

## Supported Video Formats

| Format | Extension | Recommended |
|--------|-----------|-------------|
| MP4    | .mp4      | ✅ Yes      |
| AVI    | .avi      | ✅ Yes      |
| MOV    | .mov      | ✅ Yes      |
| MKV    | .mkv      | ⚠️ Limited  |
| WMV    | .wmv      | ⚠️ Limited  |
| FLV    | .flv      | ⚠️ Limited  |
| WebM   | .webm     | ⚠️ Limited  |
| M4V    | .m4v      | ✅ Yes      |

## Example Usage

### Basic Analysis
```python
from thermal_video_processor_simple import SimpleThermalVideoProcessor

processor = SimpleThermalVideoProcessor()
result = processor.process_video(
    "path/to/video.mp4",
    analysis_mode="summary"
)

print(f"Video duration: {result['video_info']['duration']}s")
print(f"Analysis summary: {result['video_summary']}")
```

### Advanced Analysis
```python
from thermal_video_processor import ThermalVideoProcessor

processor = ThermalVideoProcessor()
result = processor.process_video(
    "path/to/video.mp4",
    custom_prompt="Analyze this thermal video for human detection and temperature anomalies.",
    analysis_mode="comprehensive"
)
```

## Testing Your Setup

Run these commands to test your video processing setup:

```bash
# Test basic video processing
python debug_video_upload.py

# Test simplified processor
python test_simple_video.py

# Test full processor (requires Hugging Face token)
python test_video_processing.py
```

## Getting Help

If you encounter issues:

1. **Check the debug output** from the test scripts
2. **Try the simplified processor** first
3. **Use smaller/shorter videos** for testing
4. **Check system resources** (RAM, disk space)
5. **Verify video file format** and codec

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid video file" | Unsupported format/corrupted file | Convert to MP4, check file integrity |
| "No valid frames extracted" | Codec issues | Use H.264 codec, try different video |
| "Analysis failed" | Memory/system issues | Reduce frame count, close other apps |
| "Processing timeout" | Large video/complex analysis | Use summary mode, shorter video |

---

**🎉 Happy Video Analysis!** 

The system will automatically fall back to simplified analysis if the advanced VLM analysis fails, ensuring you always get results. 