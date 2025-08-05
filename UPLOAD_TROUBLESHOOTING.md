# 🔧 Video Upload Troubleshooting Guide

## ❌ Problem: "video/mp4 files are not allowed" Error

### What Happened
You tried to upload an MP4 video file, but the interface showed an error saying "video/mp4 files are not allowed." This happened because the interface was still in **Image Mode** instead of **Video Mode**.

### ✅ Solution

#### Step 1: Switch to Video Mode
1. **Look at the sidebar** on the left side of the screen
2. **Find the "Input Type" section**
3. **Click on "Video"** (not "Image")
4. **Verify the mode is active** - you should see "🎬 Video Analysis Mode Active" in green

#### Step 2: Upload Your Video
1. **The interface should now show:**
   - Orange header: "🎬 Upload Thermal Video"
   - Video file uploader with supported formats
   - Analysis mode settings
   - Frame count slider

2. **Click "Browse files"** and select your MP4 video

3. **The uploader should accept:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`

### 🔍 How to Tell You're in the Right Mode

| Image Mode | Video Mode |
|------------|------------|
| 📤 Upload Thermal Image | 🎬 Upload Thermal Video |
| Blue header | Orange header |
| Image file types only | Video file types only |
| No analysis settings | Analysis mode + frame count |

### 🚨 Common Mistakes

1. **Forgot to switch modes** - Most common issue
2. **Uploaded video in Image mode** - Will show "not allowed" error
3. **Uploaded image in Video mode** - Will show "not allowed" error

### 🎯 Quick Fix Steps

1. **Stop the current upload** (if any)
2. **In the sidebar, click "Video"** under Input Type
3. **Wait for the interface to update** (should show orange header)
4. **Try uploading your video again**

### 📋 Video Upload Checklist

Before uploading, make sure:

- [ ] **Input Type** is set to "Video" in sidebar
- [ ] You see "🎬 Video Analysis Mode Active" in green
- [ ] The header says "🎬 Upload Thermal Video"
- [ ] File uploader shows video formats (mp4, avi, etc.)
- [ ] Analysis mode and frame count settings are visible

### 🆘 Still Having Issues?

If you're still getting the "not allowed" error:

1. **Refresh the page** (F5 or Ctrl+R)
2. **Clear browser cache** and try again
3. **Check the sidebar** - make sure "Video" is selected
4. **Try a different video file** to test

### 🎬 Test with Sample Video

If you want to test without uploading:

1. **Scroll down** to "📁 Or Select from Test Videos"
2. **Choose "sample_thermal.mp4"** from the dropdown
3. **Click "🎬 Analyze Selected Test Video"**

This will test the video processing without needing to upload a file.

---

**💡 Pro Tip:** Always check the sidebar first! The Input Type selection controls what file types are accepted. 