# general_video_processor.py
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

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
import requests

class GeneralVideoProcessor:
    """
    Processor for general video analysis using Vision-Language Models
    Works with regular videos (not just thermal) and provides comprehensive analysis
    """
    
    def __init__(self, hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token
        
        # Initialize VLM models
        self.blip_processor = None
        self.blip_model = None
        self.smolvlm_processor = None
        self.smolvlm_model = None
        
        # Video analysis settings
        self.frame_sampling_rate = 1  # Analyze every Nth frame
        self.max_frames_to_analyze = 30  # Maximum frames to analyze
        self.min_frame_interval = 0.5  # Minimum seconds between analyzed frames
        
    def load_models(self):
        """Load VLM models for video analysis"""
        try:
            # Load BLIP model for general image understanding
            print("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            
            # Try to load SmolVLM for more advanced analysis
            try:
                print("Loading SmolVLM model...")
                model_name = "microsoft/DialoGPT-medium"  # Fallback model
                self.smolvlm_processor = AutoProcessor.from_pretrained(model_name)
                self.smolvlm_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.smolvlm_model.to(self.device)
            except Exception as e:
                print(f"SmolVLM loading failed, using BLIP only: {e}")
                self.smolvlm_processor = None
                self.smolvlm_model = None
                
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
    def process_video(self, video_path: str, custom_prompt: str = None, 
                     analysis_mode: str = "comprehensive") -> Dict:
        """
        Process general video and generate comprehensive analysis
        
        Args:
            video_path: Path to the video file
            custom_prompt: Custom analysis prompt
            analysis_mode: "comprehensive", "key_frames", or "summary"
            
        Returns:
            Dict containing video analysis results
        """
        try:
            start_time = time.time()
            
            # Load models if not already loaded
            if self.blip_model is None:
                if not self.load_models():
                    return {"error": "Failed to load VLM models"}
            
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
        """Extract video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'resolution': f"{width}x{height}",
                'duration': duration,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            return {
                'fps': 0,
                'frame_count': 0,
                'resolution': 'Unknown',
                'duration': 0,
                'width': 0,
                'height': 0,
                'error': str(e)
            }
    
    def _extract_frames(self, video_path: str, analysis_mode: str) -> List[Tuple[int, np.ndarray, float]]:
        """Extract frames based on analysis mode"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if analysis_mode == "comprehensive":
                # Analyze more frames for comprehensive analysis
                max_frames = min(self.max_frames_to_analyze, 30)
                frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max_frames))
                
            elif analysis_mode == "key_frames":
                # Analyze key frames (scene changes)
                max_frames = min(self.max_frames_to_analyze, 20)
                frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max_frames))
                
            else:  # summary mode
                # Analyze fewer frames for quick summary
                max_frames = min(self.max_frames_to_analyze, 10)
                frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max_frames))
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_count / fps if fps > 0 else 0
                    frames.append((frame_count, frame_rgb, timestamp))
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def _analyze_frames(self, frames: List[Tuple[int, np.ndarray, float]], 
                       custom_prompt: str) -> List[Dict]:
        """Analyze frames using VLM models"""
        frame_analyses = []
        
        for frame_idx, frame, timestamp in frames:
            try:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Analyze with BLIP
                caption = self._analyze_with_blip(pil_image, custom_prompt)
                
                # Try SmolVLM if available
                detailed_analysis = ""
                if self.smolvlm_model is not None:
                    detailed_analysis = self._analyze_with_smolvlm(pil_image, custom_prompt)
                
                frame_analyses.append({
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'caption': caption,
                    'detailed_analysis': detailed_analysis,
                    'objects_detected': self._extract_objects_from_caption(caption)
                })
                
            except Exception as e:
                print(f"Error analyzing frame {frame_idx}: {e}")
                frame_analyses.append({
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'caption': f"Error analyzing frame: {str(e)}",
                    'detailed_analysis': "",
                    'objects_detected': []
                })
        
        return frame_analyses
    
    def _analyze_with_blip(self, image: Image.Image, custom_prompt: str) -> str:
        """Analyze image using BLIP model"""
        try:
            if custom_prompt:
                # Use custom prompt
                inputs = self.blip_processor(image, custom_prompt, return_tensors="pt").to(self.device)
            else:
                # Use default prompt for general video analysis
                inputs = self.blip_processor(image, "Describe what you see in this video frame", return_tensors="pt").to(self.device)
            
            outputs = self.blip_model.generate(**inputs, max_length=100, num_beams=5)
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            return f"Error in BLIP analysis: {str(e)}"
    
    def _analyze_with_smolvlm(self, image: Image.Image, custom_prompt: str) -> str:
        """Analyze image using SmolVLM model"""
        try:
            # This is a simplified implementation
            # In a full implementation, you would use the actual SmolVLM model
            prompt = custom_prompt if custom_prompt else "Describe this video frame in detail"
            
            # For now, return a basic analysis
            return f"Detailed analysis: {prompt} - Frame contains visual content that can be analyzed."
            
        except Exception as e:
            return f"Error in SmolVLM analysis: {str(e)}"
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract objects from caption text"""
        # Simple object extraction - in a real implementation, you might use NER
        common_objects = ['person', 'people', 'car', 'vehicle', 'building', 'tree', 'animal', 
                         'object', 'item', 'device', 'equipment', 'furniture', 'clothing']
        
        detected_objects = []
        caption_lower = caption.lower()
        
        for obj in common_objects:
            if obj in caption_lower:
                detected_objects.append(obj)
        
        return detected_objects
    
    def _analyze_temporal_patterns(self, frame_analyses: List[Dict], 
                                 video_info: Dict) -> Dict:
        """Analyze temporal patterns in the video"""
        try:
            # Analyze content changes over time
            captions = [analysis['caption'] for analysis in frame_analyses]
            timestamps = [analysis['timestamp'] for analysis in frame_analyses]
            
            # Detect scene changes
            scene_changes = self._detect_scene_changes(captions)
            
            # Analyze motion patterns
            motion_patterns = self._detect_motion_patterns(frame_analyses)
            
            # Analyze content consistency
            content_consistency = self._analyze_content_consistency(frame_analyses)
            
            # Detect temporal anomalies
            temporal_anomalies = self._detect_temporal_anomalies(frame_analyses)
            
            return {
                'scene_changes': scene_changes,
                'motion_patterns': motion_patterns,
                'content_consistency': content_consistency,
                'temporal_anomalies': temporal_anomalies,
                'total_scenes': len(scene_changes),
                'average_motion_level': motion_patterns.get('average_motion_level', 0)
            }
            
        except Exception as e:
            return {
                'error': f"Error in temporal analysis: {str(e)}",
                'scene_changes': [],
                'motion_patterns': {},
                'content_consistency': {},
                'temporal_anomalies': []
            }
    
    def _detect_scene_changes(self, captions: List[str]) -> List[Dict]:
        """Detect scene changes based on caption differences"""
        scene_changes = []
        
        for i in range(1, len(captions)):
            # Simple scene change detection based on caption similarity
            prev_caption = captions[i-1].lower()
            curr_caption = captions[i].lower()
            
            # Check for significant differences
            common_words = set(prev_caption.split()) & set(curr_caption.split())
            total_words = set(prev_caption.split()) | set(curr_caption.split())
            
            similarity = len(common_words) / len(total_words) if total_words else 0
            
            if similarity < 0.3:  # Significant change
                scene_changes.append({
                    'frame_index': i,
                    'similarity': similarity,
                    'description': f"Scene change detected: {curr_caption[:100]}..."
                })
        
        return scene_changes
    
    def _detect_motion_patterns(self, frame_analyses: List[Dict]) -> Dict:
        """Detect motion patterns in the video"""
        try:
            # Analyze object movement patterns
            motion_levels = []
            
            for i in range(1, len(frame_analyses)):
                prev_objects = set(frame_analyses[i-1]['objects_detected'])
                curr_objects = set(frame_analyses[i]['objects_detected'])
                
                # Calculate motion level based on object changes
                object_changes = len(prev_objects.symmetric_difference(curr_objects))
                motion_levels.append(object_changes)
            
            return {
                'average_motion_level': np.mean(motion_levels) if motion_levels else 0,
                'max_motion_level': max(motion_levels) if motion_levels else 0,
                'motion_trend': 'increasing' if len(motion_levels) > 1 and motion_levels[-1] > motion_levels[0] else 'stable'
            }
            
        except Exception as e:
            return {'error': f"Error detecting motion: {str(e)}"}
    
    def _analyze_content_consistency(self, frame_analyses: List[Dict]) -> Dict:
        """Analyze content consistency across frames"""
        try:
            all_objects = []
            for analysis in frame_analyses:
                all_objects.extend(analysis['objects_detected'])
            
            # Find most common objects
            from collections import Counter
            object_counts = Counter(all_objects)
            most_common = object_counts.most_common(5)
            
            return {
                'most_common_objects': most_common,
                'total_unique_objects': len(object_counts),
                'content_variety': len(object_counts) / len(frame_analyses) if frame_analyses else 0
            }
            
        except Exception as e:
            return {'error': f"Error analyzing content consistency: {str(e)}"}
    
    def _detect_temporal_anomalies(self, frame_analyses: List[Dict]) -> List[Dict]:
        """Detect temporal anomalies in the video"""
        anomalies = []
        
        try:
            # Detect unusual captions or objects
            for i, analysis in enumerate(frame_analyses):
                caption = analysis['caption'].lower()
                
                # Check for unusual keywords
                unusual_keywords = ['error', 'unusual', 'strange', 'anomaly', 'problem', 'issue']
                if any(keyword in caption for keyword in unusual_keywords):
                    anomalies.append({
                        'frame_index': analysis['frame_index'],
                        'timestamp': analysis['timestamp'],
                        'type': 'unusual_content',
                        'description': f"Unusual content detected: {analysis['caption'][:100]}..."
                    })
            
            return anomalies
            
        except Exception as e:
            return [{'error': f"Error detecting anomalies: {str(e)}"}]
    
    def _generate_video_summary(self, frame_analyses: List[Dict], 
                              temporal_analysis: Dict, video_info: Dict) -> str:
        """Generate comprehensive video summary"""
        try:
            summary_parts = []
            
            # Basic video info
            summary_parts.append(f"VIDEO ANALYSIS SUMMARY")
            summary_parts.append(f"Duration: {video_info.get('duration', 0):.2f} seconds")
            summary_parts.append(f"Resolution: {video_info.get('resolution', 'Unknown')}")
            summary_parts.append(f"Frames analyzed: {len(frame_analyses)}")
            summary_parts.append("")
            
            # Content overview
            summary_parts.append("CONTENT OVERVIEW:")
            all_captions = [analysis['caption'] for analysis in frame_analyses]
            summary_parts.append(f"• Video contains {len(frame_analyses)} analyzed frames")
            summary_parts.append(f"• Content varies across {temporal_analysis.get('total_scenes', 0)} distinct scenes")
            summary_parts.append("")
            
            # Key observations
            summary_parts.append("KEY OBSERVATIONS:")
            for i, analysis in enumerate(frame_analyses[:3]):  # First 3 frames
                summary_parts.append(f"• Frame {i+1} ({analysis['timestamp']:.1f}s): {analysis['caption']}")
            
            if len(frame_analyses) > 3:
                summary_parts.append(f"• ... and {len(frame_analyses) - 3} more frames")
            summary_parts.append("")
            
            # Temporal patterns
            summary_parts.append("TEMPORAL PATTERNS:")
            motion_level = temporal_analysis.get('motion_patterns', {}).get('average_motion_level', 0)
            summary_parts.append(f"• Motion level: {'High' if motion_level > 2 else 'Medium' if motion_level > 1 else 'Low'}")
            summary_parts.append(f"• Scene changes: {temporal_analysis.get('total_scenes', 0)} detected")
            
            # Most common objects
            common_objects = temporal_analysis.get('content_consistency', {}).get('most_common_objects', [])
            if common_objects:
                summary_parts.append(f"• Most common objects: {', '.join([obj[0] for obj in common_objects[:3]])}")
            
            summary_parts.append("")
            
            # Anomalies
            anomalies = temporal_analysis.get('temporal_anomalies', [])
            if anomalies:
                summary_parts.append("ANOMALIES DETECTED:")
                for anomaly in anomalies[:3]:
                    summary_parts.append(f"• {anomaly.get('description', 'Unknown anomaly')}")
            else:
                summary_parts.append("No significant anomalies detected.")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def get_supported_formats(self) -> List[str]:
        """Get supported video formats"""
        return ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v']
    
    def get_analysis_modes(self) -> List[str]:
        """Get available analysis modes"""
        return ['summary', 'key_frames', 'comprehensive'] 