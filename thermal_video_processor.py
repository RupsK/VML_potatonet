# thermal_video_processor.py
import cv2
import numpy as np
import torch
from PIL import Image
import time
from pathlib import Path
import tempfile
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from thermal_vlm_processor import ThermalImageProcessor
from thermal_smolvlm_processor import SmolVLMProcessor

class ThermalVideoProcessor:
    """
    Processor for thermal video analysis using Vision-Language Models
    Supports frame extraction, temporal analysis, and video-specific insights
    """
    
    def __init__(self, hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token
        
        # Initialize processors
        self.thermal_processor = ThermalImageProcessor()
        self.smolvlm_processor = SmolVLMProcessor(hf_token=hf_token)
        
        # Video analysis settings
        self.frame_sampling_rate = 1  # Analyze every Nth frame
        self.max_frames_to_analyze = 30  # Maximum frames to analyze
        self.min_frame_interval = 0.5  # Minimum seconds between analyzed frames
        
    def process_video(self, video_path: str, custom_prompt: str = None, 
                     analysis_mode: str = "comprehensive") -> Dict:
        """
        Process thermal video and generate comprehensive analysis
        
        Args:
            video_path: Path to the video file
            custom_prompt: Custom analysis prompt
            analysis_mode: "comprehensive", "key_frames", or "summary"
            
        Returns:
            Dict containing video analysis results
        """
        try:
            start_time = time.time()
            
            # Validate video file
            if not self._validate_video(video_path):
                return {"error": "Invalid video file or format not supported"}
            
            # Extract video metadata
            video_info = self._extract_video_metadata(video_path)
            
            # Extract frames based on analysis mode
            frames = self._extract_frames(video_path, analysis_mode)
            
            if not frames:
                return {"error": "No valid frames extracted from video"}
            
            # Analyze frames
            frame_analyses = self._analyze_frames(frames, custom_prompt)
            
            # Generate temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(frame_analyses, video_info)
            
            # Generate comprehensive video summary
            video_summary = self._generate_video_summary(frame_analyses, temporal_analysis, video_info)
            
            processing_time = time.time() - start_time
            
            return {
                'video_info': video_info,
                'frame_analyses': frame_analyses,
                'temporal_analysis': temporal_analysis,
                'video_summary': video_summary,
                'processing_time': processing_time,
                'analysis_mode': analysis_mode,
                'total_frames_analyzed': len(frames)
            }
            
        except Exception as e:
            return {"error": f"Error processing video: {str(e)}"}
    
    def _validate_video(self, video_path: str) -> bool:
        """Validate video file and check if it can be opened"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Check if we can read at least one frame
            ret, frame = cap.read()
            cap.release()
            return ret
            
        except Exception:
            return False
    
    def _extract_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata including duration, fps, resolution, etc."""
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
                'file_path': video_path,
                'file_name': Path(video_path).name
            }
            
        except Exception as e:
            return {"error": f"Error extracting metadata: {str(e)}"}
    
    def _extract_frames(self, video_path: str, analysis_mode: str) -> List[Tuple[int, np.ndarray, float]]:
        """
        Extract frames from video based on analysis mode
        
        Returns:
            List of tuples: (frame_number, frame_array, timestamp)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            last_analyzed_time = -self.min_frame_interval
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                
                # Determine if this frame should be analyzed
                should_analyze = False
                
                if analysis_mode == "comprehensive":
                    # Analyze every Nth frame
                    should_analyze = (frame_count % self.frame_sampling_rate == 0 and 
                                    timestamp - last_analyzed_time >= self.min_frame_interval)
                elif analysis_mode == "key_frames":
                    # Analyze key frames (scene changes, significant motion)
                    should_analyze = self._is_key_frame(frame, frames)
                elif analysis_mode == "summary":
                    # Analyze evenly distributed frames
                    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    target_interval = total_duration / min(self.max_frames_to_analyze, 10)
                    should_analyze = (timestamp - last_analyzed_time >= target_interval)
                
                if should_analyze and len(frames) < self.max_frames_to_analyze:
                    frames.append((frame_count, frame, timestamp))
                    last_analyzed_time = timestamp
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def _is_key_frame(self, current_frame: np.ndarray, previous_frames: List) -> bool:
        """Detect if current frame is a key frame based on visual changes"""
        if not previous_frames:
            return True
        
        # Simple motion detection using frame difference
        last_frame = previous_frames[-1][1]  # Get the frame array
        
        # Convert to grayscale for comparison
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(current_gray, last_gray)
        motion_score = np.mean(diff)
        
        # Consider it a key frame if motion is significant
        return motion_score > 10  # Threshold for motion detection
    
    def _analyze_frames(self, frames: List[Tuple[int, np.ndarray, float]], 
                       custom_prompt: str) -> List[Dict]:
        """Analyze individual frames using thermal analysis"""
        frame_analyses = []
        
        default_prompt = "Analyze this thermal video frame. Describe what you see, including temperature patterns, objects, motion, and any anomalies."
        prompt = custom_prompt if custom_prompt else default_prompt
        
        for frame_num, frame, timestamp in frames:
            try:
                # Save frame temporarily
                temp_frame_path = f"temp_frame_{frame_num}.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                # Analyze frame using SmolVLM (video-capable model)
                frame_analysis = self.smolvlm_processor.analyze_thermal_image(
                    temp_frame_path, prompt
                )
                
                if frame_analysis:
                    frame_analysis['frame_number'] = frame_num
                    frame_analysis['timestamp'] = timestamp
                    frame_analysis['frame_path'] = temp_frame_path
                    frame_analyses.append(frame_analysis)
                
                # Clean up temp file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                    
            except Exception as e:
                print(f"Error analyzing frame {frame_num}: {e}")
                continue
        
        return frame_analyses
    
    def _analyze_temporal_patterns(self, frame_analyses: List[Dict], 
                                 video_info: Dict) -> Dict:
        """Analyze temporal patterns across frames"""
        if not frame_analyses:
            return {"error": "No frame analyses available"}
        
        try:
            # Extract temperature trends
            temperatures = [analysis['temperature_analysis']['mean_temperature'] 
                          for analysis in frame_analyses]
            timestamps = [analysis['timestamp'] for analysis in frame_analyses]
            
            # Calculate temperature trends
            temp_trend = self._calculate_temperature_trend(temperatures, timestamps)
            
            # Detect motion patterns
            motion_patterns = self._detect_motion_patterns(frame_analyses)
            
            # Analyze content consistency
            content_consistency = self._analyze_content_consistency(frame_analyses)
            
            # Detect anomalies over time
            temporal_anomalies = self._detect_temporal_anomalies(frame_analyses)
            
            return {
                'temperature_trend': temp_trend,
                'motion_patterns': motion_patterns,
                'content_consistency': content_consistency,
                'temporal_anomalies': temporal_anomalies,
                'temperature_stats': {
                    'mean': np.mean(temperatures),
                    'std': np.std(temperatures),
                    'min': np.min(temperatures),
                    'max': np.max(temperatures),
                    'range': np.max(temperatures) - np.min(temperatures)
                }
            }
            
        except Exception as e:
            return {"error": f"Error in temporal analysis: {str(e)}"}
    
    def _calculate_temperature_trend(self, temperatures: List[float], 
                                   timestamps: List[float]) -> Dict:
        """Calculate temperature trend over time"""
        if len(temperatures) < 2:
            return {"trend": "insufficient_data", "slope": 0, "correlation": 0}
        
        # Calculate linear trend
        slope = np.polyfit(timestamps, temperatures, 1)[0]
        
        # Calculate correlation
        correlation = np.corrcoef(timestamps, temperatures)[0, 1]
        
        # Determine trend direction
        if abs(slope) < 0.1:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": slope,
            "correlation": correlation,
            "temperature_change": temperatures[-1] - temperatures[0]
        }
    
    def _detect_motion_patterns(self, frame_analyses: List[Dict]) -> Dict:
        """Detect motion patterns from frame analyses"""
        if len(frame_analyses) < 2:
            return {"motion_detected": False}
        
        # Analyze captions for motion indicators
        motion_keywords = ['moving', 'motion', 'walking', 'running', 'flowing', 
                          'changing', 'dynamic', 'animation', 'movement']
        
        motion_mentions = 0
        for analysis in frame_analyses:
            caption = analysis['caption'].lower()
            if any(keyword in caption for keyword in motion_keywords):
                motion_mentions += 1
        
        motion_percentage = (motion_mentions / len(frame_analyses)) * 100
        
        return {
            "motion_detected": motion_percentage > 20,
            "motion_percentage": motion_percentage,
            "motion_mentions": motion_mentions
        }
    
    def _analyze_content_consistency(self, frame_analyses: List[Dict]) -> Dict:
        """Analyze consistency of content across frames"""
        if len(frame_analyses) < 2:
            return {"consistency": "insufficient_data"}
        
        # Extract key terms from captions
        captions = [analysis['caption'] for analysis in frame_analyses]
        
        # Simple consistency measure based on common terms
        all_words = []
        for caption in captions:
            words = caption.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Find common terms (appearing in multiple frames)
        common_terms = {word: count for word, count in word_counts.items() 
                       if count > 1 and len(word) > 3}
        
        consistency_score = len(common_terms) / len(word_counts) if word_counts else 0
        
        return {
            "consistency_score": consistency_score,
            "common_terms": list(common_terms.keys())[:10],  # Top 10 common terms
            "total_unique_terms": len(word_counts)
        }
    
    def _detect_temporal_anomalies(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect anomalies that occur over time"""
        anomalies = []
        
        if len(frame_analyses) < 3:
            return anomalies
        
        # Analyze temperature anomalies
        temperatures = [analysis['temperature_analysis']['mean_temperature'] 
                       for analysis in frame_analyses]
        
        mean_temp = np.mean(temperatures)
        std_temp = np.std(temperatures)
        
        for i, (analysis, temp) in enumerate(zip(frame_analyses, temperatures)):
            # Detect temperature spikes
            if abs(temp - mean_temp) > 2 * std_temp:
                anomalies.append({
                    'type': 'temperature_spike',
                    'frame_number': analysis['frame_number'],
                    'timestamp': analysis['timestamp'],
                    'severity': abs(temp - mean_temp) / std_temp,
                    'description': f"Temperature spike detected at frame {analysis['frame_number']}"
                })
        
        # Detect content anomalies (sudden changes in description)
        for i in range(1, len(frame_analyses)):
            prev_caption = frame_analyses[i-1]['caption']
            curr_caption = frame_analyses[i]['caption']
            
            # Simple similarity check (can be improved with embeddings)
            prev_words = set(prev_caption.lower().split())
            curr_words = set(curr_caption.lower().split())
            
            similarity = len(prev_words.intersection(curr_words)) / len(prev_words.union(curr_words))
            
            if similarity < 0.3:  # Low similarity indicates content change
                anomalies.append({
                    'type': 'content_change',
                    'frame_number': frame_analyses[i]['frame_number'],
                    'timestamp': frame_analyses[i]['timestamp'],
                    'similarity': similarity,
                    'description': f"Significant content change detected at frame {frame_analyses[i]['frame_number']}"
                })
        
        return anomalies
    
    def _generate_video_summary(self, frame_analyses: List[Dict], 
                              temporal_analysis: Dict, video_info: Dict) -> str:
        """Generate comprehensive video summary"""
        if not frame_analyses:
            return "No frames were successfully analyzed."
        
        summary_parts = []
        
        # Video overview
        summary_parts.append(f"Video Analysis Summary:")
        summary_parts.append(f"Duration: {video_info.get('duration', 0):.2f} seconds")
        summary_parts.append(f"Resolution: {video_info.get('resolution', 'Unknown')}")
        summary_parts.append(f"Frames analyzed: {len(frame_analyses)}")
        
        # Temperature analysis
        if 'temperature_stats' in temporal_analysis:
            temp_stats = temporal_analysis['temperature_stats']
            summary_parts.append(f"\nTemperature Analysis:")
            summary_parts.append(f"Average temperature: {temp_stats['mean']:.2f}")
            summary_parts.append(f"Temperature range: {temp_stats['range']:.2f}")
            summary_parts.append(f"Temperature variation: {temp_stats['std']:.2f}")
        
        # Temperature trend
        if 'temperature_trend' in temporal_analysis:
            trend = temporal_analysis['temperature_trend']
            summary_parts.append(f"\nTemperature Trend:")
            summary_parts.append(f"Overall trend: {trend['trend']}")
            if trend['trend'] != 'insufficient_data':
                summary_parts.append(f"Temperature change: {trend['temperature_change']:.2f}")
        
        # Motion analysis
        if 'motion_patterns' in temporal_analysis:
            motion = temporal_analysis['motion_patterns']
            summary_parts.append(f"\nMotion Analysis:")
            summary_parts.append(f"Motion detected: {'Yes' if motion['motion_detected'] else 'No'}")
            summary_parts.append(f"Motion percentage: {motion['motion_percentage']:.1f}%")
        
        # Content consistency
        if 'content_consistency' in temporal_analysis:
            consistency = temporal_analysis['content_consistency']
            summary_parts.append(f"\nContent Consistency:")
            summary_parts.append(f"Consistency score: {consistency['consistency_score']:.2f}")
            if consistency['common_terms']:
                summary_parts.append(f"Common elements: {', '.join(consistency['common_terms'][:5])}")
        
        # Anomalies
        if 'temporal_anomalies' in temporal_analysis and temporal_analysis['temporal_anomalies']:
            anomalies = temporal_analysis['temporal_anomalies']
            summary_parts.append(f"\nDetected Anomalies:")
            for anomaly in anomalies[:3]:  # Show top 3 anomalies
                summary_parts.append(f"- {anomaly['description']}")
        
        # Key insights from frame analyses
        summary_parts.append(f"\nKey Insights:")
        
        # Most common objects/patterns
        all_captions = [analysis['caption'] for analysis in frame_analyses]
        common_objects = self._extract_common_objects(all_captions)
        if common_objects:
            summary_parts.append(f"Common elements: {', '.join(common_objects)}")
        
        # Overall assessment
        summary_parts.append(f"\nOverall Assessment:")
        if len(frame_analyses) > 0:
            avg_confidence = np.mean([analysis.get('confidence', 0.5) for analysis in frame_analyses])
            summary_parts.append(f"Analysis confidence: {avg_confidence:.2f}")
        
        return "\n".join(summary_parts)
    
    def _extract_common_objects(self, captions: List[str]) -> List[str]:
        """Extract common objects/patterns from captions"""
        # Simple keyword extraction (can be improved with NLP)
        object_keywords = ['person', 'human', 'object', 'heat', 'cold', 'pattern', 
                          'area', 'region', 'source', 'device', 'equipment']
        
        object_counts = {}
        for caption in captions:
            caption_lower = caption.lower()
            for keyword in object_keywords:
                if keyword in caption_lower:
                    object_counts[keyword] = object_counts.get(keyword, 0) + 1
        
        # Return most common objects
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        return [obj for obj, count in sorted_objects[:5] if count > 1]
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    def get_analysis_modes(self) -> List[str]:
        """Get available analysis modes"""
        return ["comprehensive", "key_frames", "summary"] 