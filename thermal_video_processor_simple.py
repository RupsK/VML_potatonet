# thermal_video_processor_simple.py
"""
Simplified thermal video processor with better error handling
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class SimpleThermalVideoProcessor:
    """
    Simplified processor for thermal video analysis
    """
    
    def __init__(self):
        self.max_frames_to_analyze = 10  # Reduced for faster processing
        self.min_frame_interval = 1.0  # Increased interval
        
    def process_video(self, video_path: str, custom_prompt: str = None, 
                     analysis_mode: str = "summary") -> Dict:
        """
        Process thermal video with simplified analysis
        """
        try:
            start_time = time.time()
            
            # Validate video file
            if not self._validate_video(video_path):
                return {"error": "Invalid video file"}
            
            # Extract video metadata
            video_info = self._extract_video_metadata(video_path)
            
            # Extract frames (simplified)
            frames = self._extract_frames_simple(video_path)
            
            if not frames:
                return {"error": "No valid frames extracted"}
            
            # Analyze frames (simplified)
            frame_analyses = self._analyze_frames_simple(frames, custom_prompt)
            
            # Generate simple summary
            video_summary = self._generate_simple_summary(frame_analyses, video_info)
            
            processing_time = time.time() - start_time
            
            return {
                'video_info': video_info,
                'frame_analyses': frame_analyses,
                'video_summary': video_summary,
                'processing_time': processing_time,
                'analysis_mode': analysis_mode,
                'total_frames_analyzed': len(frames)
            }
            
        except Exception as e:
            return {"error": f"Error processing video: {str(e)}"}
    
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
        """Extract basic video metadata"""
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
            return {"error": f"Error extracting metadata: {str(e)}"}
    
    def _extract_frames_simple(self, video_path: str) -> List[Tuple[int, np.ndarray, float]]:
        """Extract frames with simplified logic"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            
            # Get total frames and duration
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame intervals
            if total_frames <= self.max_frames_to_analyze:
                # If video is short, analyze all frames
                frame_interval = 1
            else:
                # Otherwise, sample evenly
                frame_interval = max(1, total_frames // self.max_frames_to_analyze)
            
            while len(frames) < self.max_frames_to_analyze:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only analyze every Nth frame
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frames.append((frame_count, frame, timestamp))
                
                frame_count += 1
                
                # Safety check
                if frame_count > total_frames:
                    break
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def _analyze_frames_simple(self, frames: List[Tuple[int, np.ndarray, float]], 
                             custom_prompt: str) -> List[Dict]:
        """Analyze frames with simplified analysis"""
        frame_analyses = []
        
        for frame_num, frame, timestamp in frames:
            try:
                # Simple temperature analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate basic statistics
                mean_temp = np.mean(gray)
                max_temp = np.max(gray)
                min_temp = np.min(gray)
                std_temp = np.std(gray)
                
                # Simple caption generation based on visual analysis
                caption = self._generate_simple_caption(gray, mean_temp, max_temp, min_temp)
                
                frame_analysis = {
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'caption': caption,
                    'temperature_analysis': {
                        'mean_temperature': mean_temp,
                        'max_temperature': max_temp,
                        'min_temperature': min_temp,
                        'temperature_std': std_temp,
                        'hot_regions_percentage': np.sum(gray > mean_temp + std_temp) / gray.size * 100,
                        'cold_regions_percentage': np.sum(gray < mean_temp - std_temp) / gray.size * 100,
                        'temperature_range': max_temp - min_temp
                    },
                    'processing_time': 0.1  # Estimated
                }
                
                frame_analyses.append(frame_analysis)
                
            except Exception as e:
                print(f"Error analyzing frame {frame_num}: {e}")
                continue
        
        return frame_analyses
    
    def _generate_simple_caption(self, gray_frame: np.ndarray, mean_temp: float, 
                               max_temp: float, min_temp: float) -> str:
        """Generate simple caption based on visual analysis"""
        try:
            # Analyze brightness patterns
            bright_pixels = np.sum(gray_frame > mean_temp + 30)
            dark_pixels = np.sum(gray_frame < mean_temp - 30)
            total_pixels = gray_frame.size
            
            bright_percent = (bright_pixels / total_pixels) * 100
            dark_percent = (dark_pixels / total_pixels) * 100
            
            # Generate descriptive caption
            caption_parts = []
            
            if bright_percent > 20:
                caption_parts.append("High temperature regions detected")
            elif dark_percent > 20:
                caption_parts.append("Low temperature regions detected")
            else:
                caption_parts.append("Moderate temperature distribution")
            
            # Add temperature range info
            temp_range = max_temp - min_temp
            if temp_range > 100:
                caption_parts.append("with significant temperature variation")
            elif temp_range > 50:
                caption_parts.append("with moderate temperature variation")
            else:
                caption_parts.append("with minimal temperature variation")
            
            # Add motion detection (simple edge analysis)
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1:
                caption_parts.append("showing detailed thermal patterns")
            else:
                caption_parts.append("showing uniform thermal patterns")
            
            return ". ".join(caption_parts) + "."
            
        except Exception as e:
            return f"Thermal frame analysis: Mean temperature {mean_temp:.1f}, range {max_temp - min_temp:.1f}"
    
    def _generate_simple_summary(self, frame_analyses: List[Dict], video_info: Dict) -> str:
        """Generate simple video summary"""
        if not frame_analyses:
            return "No frames were successfully analyzed."
        
        summary_parts = []
        
        # Video overview
        summary_parts.append(f"Video Analysis Summary:")
        summary_parts.append(f"Duration: {video_info.get('duration', 0):.2f} seconds")
        summary_parts.append(f"Resolution: {video_info.get('resolution', 'Unknown')}")
        summary_parts.append(f"Frames analyzed: {len(frame_analyses)}")
        
        # Temperature analysis
        if frame_analyses:
            temperatures = [analysis['temperature_analysis']['mean_temperature'] 
                          for analysis in frame_analyses]
            
            summary_parts.append(f"\nTemperature Analysis:")
            summary_parts.append(f"Average temperature: {np.mean(temperatures):.2f}")
            summary_parts.append(f"Temperature range: {np.max(temperatures) - np.min(temperatures):.2f}")
            summary_parts.append(f"Temperature variation: {np.std(temperatures):.2f}")
        
        # Content summary
        summary_parts.append(f"\nContent Analysis:")
        summary_parts.append(f"Thermal patterns detected across {len(frame_analyses)} frames")
        
        # Detect common themes
        captions = [analysis['caption'] for analysis in frame_analyses]
        if any("High temperature" in caption for caption in captions):
            summary_parts.append("High temperature regions observed")
        if any("Low temperature" in caption for caption in captions):
            summary_parts.append("Low temperature regions observed")
        if any("significant temperature variation" in caption for caption in captions):
            summary_parts.append("Significant temperature variations detected")
        
        return "\n".join(summary_parts)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    def get_analysis_modes(self) -> List[str]:
        """Get available analysis modes"""
        return ["summary", "key_frames", "comprehensive"] 