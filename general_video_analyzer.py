#!/usr/bin/env python3
"""
General Video Analyzer
For regular video analysis (not thermal)
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class GeneralVideoAnalyzer:
    """
    General video analyzer for regular videos
    Focuses on motion detection, object tracking, and scene analysis
    """
    
    def __init__(self):
        self.max_frames_to_analyze = 10
        self.min_frame_interval = 1.0
        
    def analyze_video(self, video_path: str, analysis_mode: str = "summary") -> Dict:
        """
        Analyze regular video for motion, objects, and scene changes
        """
        try:
            start_time = time.time()
            
            # Validate video file
            if not self._validate_video(video_path):
                return {"error": "Invalid video file"}
            
            # Extract video metadata
            video_info = self._extract_video_metadata(video_path)
            
            # Extract frames
            frames = self._extract_frames(video_path)
            
            if not frames:
                return {"error": "No valid frames extracted"}
            
            # Analyze frames
            frame_analyses = self._analyze_frames(frames)
            
            # Generate motion analysis
            motion_analysis = self._analyze_motion_patterns(frame_analyses, video_info)
            
            # Generate scene analysis
            scene_analysis = self._analyze_scene_changes(frame_analyses)
            
            # Generate summary
            video_summary = self._generate_summary(frame_analyses, motion_analysis, scene_analysis, video_info)
            
            processing_time = time.time() - start_time
            
            return {
                'video_info': video_info,
                'frame_analyses': frame_analyses,
                'motion_analysis': motion_analysis,
                'scene_analysis': scene_analysis,
                'video_summary': video_summary,
                'processing_time': processing_time,
                'analysis_mode': analysis_mode,
                'total_frames_analyzed': len(frames)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing video: {str(e)}"}
    
    def _validate_video(self, video_path: str) -> bool:
        """Validate video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            ret, frame = cap.read()
            cap.release()
            return ret
            
        except Exception:
            return False
    
    def _extract_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'file_name': Path(video_path).name
            }
        except Exception as e:
            return {
                'fps': 0,
                'frame_count': 0,
                'duration': 0,
                'width': 0,
                'height': 0,
                'resolution': 'Unknown',
                'file_name': Path(video_path).name,
                'error': str(e)
            }
    
    def _extract_frames(self, video_path: str) -> List[Tuple[int, np.ndarray, float]]:
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while len(frames) < self.max_frames_to_analyze:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on interval
                if frame_count % max(1, int(fps * self.min_frame_interval)) == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    frames.append((frame_count, frame, timestamp))
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def _analyze_frames(self, frames: List[Tuple[int, np.ndarray, float]]) -> List[Dict]:
        """Analyze individual frames"""
        analyses = []
        
        for frame_number, frame, timestamp in frames:
            try:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Basic frame analysis
                analysis = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'brightness': np.mean(gray),
                    'contrast': np.std(gray),
                    'motion_score': 0,  # Will be calculated later
                    'description': self._generate_frame_description(frame, gray),
                    'processing_time': 0
                }
                
                analyses.append(analysis)
                
            except Exception as e:
                print(f"Error analyzing frame {frame_number}: {e}")
                continue
        
        return analyses
    
    def _generate_frame_description(self, frame: np.ndarray, gray: np.ndarray) -> str:
        """Generate description for a frame"""
        try:
            # Basic image statistics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Simple scene classification
            if brightness < 50:
                scene_type = "dark scene"
            elif brightness > 200:
                scene_type = "bright scene"
            else:
                scene_type = "normal lighting"
            
            if contrast < 20:
                scene_detail = "low contrast"
            elif contrast > 60:
                scene_detail = "high contrast"
            else:
                scene_detail = "moderate contrast"
            
            # Color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            
            if saturation < 50:
                color_desc = "mostly grayscale"
            elif saturation > 150:
                color_desc = "highly saturated"
            else:
                color_desc = "moderate color"
            
            return f"Frame shows {scene_type} with {scene_detail} and {color_desc}. Brightness: {brightness:.1f}, Contrast: {contrast:.1f}"
            
        except Exception as e:
            return f"Frame analysis error: {str(e)}"
    
    def _analyze_motion_patterns(self, frame_analyses: List[Dict], video_info: Dict) -> Dict:
        """Analyze motion patterns across frames"""
        try:
            if len(frame_analyses) < 2:
                return {"motion_detected": False, "motion_percentage": 0}
            
            # Calculate motion between consecutive frames
            motion_scores = []
            for i in range(1, len(frame_analyses)):
                # Simple motion detection (can be enhanced)
                motion_score = abs(frame_analyses[i]['brightness'] - frame_analyses[i-1]['brightness'])
                motion_scores.append(motion_score)
                frame_analyses[i]['motion_score'] = motion_score
            
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            motion_detected = avg_motion > 5  # Threshold for motion detection
            
            return {
                "motion_detected": motion_detected,
                "motion_percentage": min(100, avg_motion * 10),
                "average_motion_score": avg_motion,
                "motion_variance": np.var(motion_scores) if motion_scores else 0
            }
            
        except Exception as e:
            return {"motion_detected": False, "motion_percentage": 0, "error": str(e)}
    
    def _analyze_scene_changes(self, frame_analyses: List[Dict]) -> Dict:
        """Analyze scene changes and patterns"""
        try:
            if len(frame_analyses) < 2:
                return {"scene_changes": 0, "consistency_score": 1.0}
            
            # Detect significant scene changes
            scene_changes = 0
            brightness_values = [analysis['brightness'] for analysis in frame_analyses]
            
            for i in range(1, len(brightness_values)):
                change = abs(brightness_values[i] - brightness_values[i-1])
                if change > 30:  # Threshold for scene change
                    scene_changes += 1
            
            # Calculate consistency
            consistency_score = 1.0 - (scene_changes / len(frame_analyses))
            
            return {
                "scene_changes": scene_changes,
                "consistency_score": consistency_score,
                "brightness_trend": "stable" if consistency_score > 0.8 else "variable"
            }
            
        except Exception as e:
            return {"scene_changes": 0, "consistency_score": 1.0, "error": str(e)}
    
    def _generate_summary(self, frame_analyses: List[Dict], motion_analysis: Dict, 
                         scene_analysis: Dict, video_info: Dict) -> str:
        """Generate comprehensive video summary"""
        try:
            summary_parts = []
            
            # Video info
            summary_parts.append(f"Video Analysis Summary:")
            summary_parts.append(f"File: {video_info['file_name']}")
            summary_parts.append(f"Duration: {video_info['duration']:.2f} seconds")
            summary_parts.append(f"Resolution: {video_info['resolution']}")
            summary_parts.append(f"FPS: {video_info['fps']:.1f}")
            summary_parts.append(f"Frames analyzed: {len(frame_analyses)}")
            
            # Motion analysis
            if motion_analysis.get('motion_detected'):
                summary_parts.append(f"\nMotion Analysis:")
                summary_parts.append(f"- Motion detected: Yes")
                summary_parts.append(f"- Motion intensity: {motion_analysis['motion_percentage']:.1f}%")
                summary_parts.append(f"- Average motion score: {motion_analysis['average_motion_score']:.2f}")
            else:
                summary_parts.append(f"\nMotion Analysis:")
                summary_parts.append(f"- Motion detected: No (static scene)")
            
            # Scene analysis
            summary_parts.append(f"\nScene Analysis:")
            summary_parts.append(f"- Scene changes: {scene_analysis.get('scene_changes', 0)}")
            summary_parts.append(f"- Consistency: {scene_analysis.get('consistency_score', 1.0):.2f}")
            summary_parts.append(f"- Brightness trend: {scene_analysis.get('brightness_trend', 'unknown')}")
            
            # Frame statistics
            if frame_analyses:
                avg_brightness = np.mean([f['brightness'] for f in frame_analyses])
                avg_contrast = np.mean([f['contrast'] for f in frame_analyses])
                summary_parts.append(f"\nFrame Statistics:")
                summary_parts.append(f"- Average brightness: {avg_brightness:.1f}")
                summary_parts.append(f"- Average contrast: {avg_contrast:.1f}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"

def test_general_analyzer():
    """Test the general video analyzer"""
    print("🎬 Testing General Video Analyzer")
    print("=" * 50)
    
    # Find test videos
    test_folder = "test_video"
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found")
        return False
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(Path(test_folder).glob(f"*{ext}"))
    
    if not video_files:
        print("❌ No video files found")
        return False
    
    # Use the smallest file
    test_video = min(video_files, key=lambda x: x.stat().st_size)
    print(f"📹 Testing with: {test_video.name} ({test_video.stat().st_size / 1024:.1f} KB)")
    
    try:
        analyzer = GeneralVideoAnalyzer()
        analyzer.max_frames_to_analyze = 5  # Small test
        
        start_time = time.time()
        result = analyzer.analyze_video(str(test_video), "summary")
        processing_time = time.time() - start_time
        
        if result and 'error' not in result:
            print("✅ General analyzer test PASSED")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Frames analyzed: {result['total_frames_analyzed']}")
            print(f"   - Motion detected: {result['motion_analysis']['motion_detected']}")
            print(f"   - Scene changes: {result['scene_analysis']['scene_changes']}")
            return True
        else:
            print(f"❌ General analyzer test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    test_general_analyzer() 