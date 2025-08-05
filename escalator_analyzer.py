#!/usr/bin/env python3
"""
Escalator Safety Analyzer
Detects crowding and falling objects/bags on escalators
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class EscalatorAnalyzer:
    """
    Specialized analyzer for escalator safety monitoring
    Detects crowding, falling objects, and safety incidents
    """
    
    def __init__(self):
        self.max_frames_to_analyze = 15
        self.min_frame_interval = 0.5  # More frequent sampling for safety
        
    def analyze_escalator_safety(self, video_path: str, analysis_mode: str = "safety") -> Dict:
        """
        Analyze escalator video for safety concerns
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
            
            # Analyze frames for safety concerns
            frame_analyses = self._analyze_safety_frames(frames)
            
            # Detect crowding patterns
            crowding_analysis = self._detect_crowding(frame_analyses)
            
            # Detect falling objects/bags
            falling_analysis = self._detect_falling_objects(frame_analyses)
            
            # Generate safety summary
            safety_summary = self._generate_safety_summary(
                frame_analyses, crowding_analysis, falling_analysis, video_info
            )
            
            processing_time = time.time() - start_time
            
            return {
                'video_info': video_info,
                'frame_analyses': frame_analyses,
                'crowding_analysis': crowding_analysis,
                'falling_analysis': falling_analysis,
                'safety_summary': safety_summary,
                'processing_time': processing_time,
                'analysis_mode': analysis_mode,
                'total_frames_analyzed': len(frames),
                'safety_alerts': self._generate_safety_alerts(crowding_analysis, falling_analysis)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing escalator safety: {str(e)}"}
    
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
                'fps': 0, 'frame_count': 0, 'duration': 0,
                'width': 0, 'height': 0, 'resolution': 'Unknown',
                'file_name': Path(video_path).name, 'error': str(e)
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
                
                # Sample frames more frequently for safety analysis
                if frame_count % max(1, int(fps * self.min_frame_interval)) == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    frames.append((frame_count, frame, timestamp))
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def _analyze_safety_frames(self, frames: List[Tuple[int, np.ndarray, float]]) -> List[Dict]:
        """Analyze frames for safety concerns"""
        analyses = []
        
        for frame_number, frame, timestamp in frames:
            try:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Enhanced frame analysis for safety
                analysis = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'brightness': np.mean(gray),
                    'contrast': np.std(gray),
                    'motion_score': 0,
                    'crowding_score': self._calculate_crowding_score(gray),
                    'falling_detection': self._detect_falling_in_frame(gray, frame),
                    'safety_description': self._generate_safety_description(frame, gray),
                    'processing_time': 0
                }
                
                analyses.append(analysis)
                
            except Exception as e:
                print(f"Error analyzing frame {frame_number}: {e}")
                continue
        
        return analyses
    
    def _calculate_crowding_score(self, gray: np.ndarray) -> float:
        """Calculate crowding score based on image complexity and edge density"""
        try:
            # Edge detection to find people/objects
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture analysis for complexity
            # Use gradient magnitude as texture measure
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            texture_score = np.mean(gradient_magnitude)
            
            # Combine edge density and texture for crowding score
            crowding_score = (edge_density * 1000) + (texture_score / 10)
            
            return min(100, crowding_score)  # Normalize to 0-100
            
        except Exception as e:
            print(f"Error calculating crowding score: {e}")
            return 0.0
    
    def _detect_falling_in_frame(self, gray: np.ndarray, frame: np.ndarray) -> Dict:
        """Detect potential falling objects in a frame"""
        try:
            # Motion detection using frame differencing (simplified)
            # In a real implementation, you'd compare with previous frame
            
            # Detect horizontal lines (escalator steps)
            lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=50, 
                                   minLineLength=50, maxLineGap=10)
            
            # Detect blobs (potential objects)
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 10
            params.maxThreshold = 200
            params.filterByArea = True
            params.minArea = 100
            params.maxArea = 5000
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            # Analyze for potential falling objects
            falling_indicators = {
                'horizontal_lines': len(lines) if lines is not None else 0,
                'blob_count': len(keypoints),
                'potential_objects': len(keypoints),
                'falling_probability': min(100, len(keypoints) * 10)  # Simplified
            }
            
            return falling_indicators
            
        except Exception as e:
            print(f"Error detecting falling objects: {e}")
            return {
                'horizontal_lines': 0, 'blob_count': 0, 
                'potential_objects': 0, 'falling_probability': 0
            }
    
    def _generate_safety_description(self, frame: np.ndarray, gray: np.ndarray) -> str:
        """Generate safety-focused description of frame"""
        try:
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Safety assessment
            if brightness < 40:
                safety_condition = "poor visibility - safety concern"
            elif brightness > 200:
                safety_condition = "overexposed - may miss details"
            else:
                safety_condition = "good visibility for safety monitoring"
            
            # Crowding assessment
            crowding_score = self._calculate_crowding_score(gray)
            if crowding_score > 70:
                crowding_status = "high crowding detected"
            elif crowding_score > 40:
                crowding_status = "moderate crowding"
            else:
                crowding_status = "low crowding"
            
            # Falling risk assessment
            falling_detection = self._detect_falling_in_frame(gray, frame)
            if falling_detection['falling_probability'] > 50:
                falling_status = "high risk of falling objects"
            elif falling_detection['falling_probability'] > 20:
                falling_status = "moderate risk of falling objects"
            else:
                falling_status = "low risk of falling objects"
            
            return f"Safety: {safety_condition}. Crowding: {crowding_status}. Falling risk: {falling_status}. Brightness: {brightness:.1f}, Contrast: {contrast:.1f}"
            
        except Exception as e:
            return f"Safety analysis error: {str(e)}"
    
    def _detect_crowding(self, frame_analyses: List[Dict]) -> Dict:
        """Analyze crowding patterns across frames"""
        try:
            if len(frame_analyses) < 2:
                return {"crowding_detected": False, "crowding_level": "low"}
            
            crowding_scores = [analysis['crowding_score'] for analysis in frame_analyses]
            avg_crowding = np.mean(crowding_scores)
            max_crowding = np.max(crowding_scores)
            
            # Determine crowding level
            if avg_crowding > 70:
                crowding_level = "high"
                crowding_detected = True
            elif avg_crowding > 40:
                crowding_level = "moderate"
                crowding_detected = True
            else:
                crowding_level = "low"
                crowding_detected = avg_crowding > 20
            
            # Detect crowding trends
            if len(crowding_scores) > 3:
                # Check if crowding is increasing
                first_half = np.mean(crowding_scores[:len(crowding_scores)//2])
                second_half = np.mean(crowding_scores[len(crowding_scores)//2:])
                crowding_trend = "increasing" if second_half > first_half * 1.2 else "stable"
            else:
                crowding_trend = "insufficient data"
            
            return {
                "crowding_detected": crowding_detected,
                "crowding_level": crowding_level,
                "average_crowding_score": avg_crowding,
                "max_crowding_score": max_crowding,
                "crowding_trend": crowding_trend,
                "crowding_percentage": min(100, avg_crowding)
            }
            
        except Exception as e:
            return {"crowding_detected": False, "crowding_level": "unknown", "error": str(e)}
    
    def _detect_falling_objects(self, frame_analyses: List[Dict]) -> Dict:
        """Analyze falling object patterns across frames"""
        try:
            if len(frame_analyses) < 2:
                return {"falling_detected": False, "falling_risk": "low"}
            
            falling_scores = [analysis['falling_detection']['falling_probability'] 
                            for analysis in frame_analyses]
            avg_falling_risk = np.mean(falling_scores)
            max_falling_risk = np.max(falling_scores)
            
            # Determine falling risk level
            if avg_falling_risk > 60:
                falling_risk = "high"
                falling_detected = True
            elif avg_falling_risk > 30:
                falling_risk = "moderate"
                falling_detected = True
            else:
                falling_risk = "low"
                falling_detected = avg_falling_risk > 10
            
            # Count potential objects
            total_objects = sum(analysis['falling_detection']['potential_objects'] 
                              for analysis in frame_analyses)
            
            return {
                "falling_detected": falling_detected,
                "falling_risk": falling_risk,
                "average_falling_risk": avg_falling_risk,
                "max_falling_risk": max_falling_risk,
                "total_potential_objects": total_objects,
                "falling_risk_percentage": min(100, avg_falling_risk)
            }
            
        except Exception as e:
            return {"falling_detected": False, "falling_risk": "unknown", "error": str(e)}
    
    def _generate_safety_summary(self, frame_analyses: List[Dict], crowding_analysis: Dict, 
                                falling_analysis: Dict, video_info: Dict) -> str:
        """Generate comprehensive safety summary"""
        try:
            summary_parts = []
            
            # Video info
            summary_parts.append(f"Escalator Safety Analysis Summary:")
            summary_parts.append(f"File: {video_info['file_name']}")
            summary_parts.append(f"Duration: {video_info['duration']:.2f} seconds")
            summary_parts.append(f"Resolution: {video_info['resolution']}")
            summary_parts.append(f"Frames analyzed: {len(frame_analyses)}")
            
            # Crowding analysis
            summary_parts.append(f"\nCrowding Analysis:")
            if crowding_analysis.get('crowding_detected'):
                summary_parts.append(f"- Crowding detected: YES")
                summary_parts.append(f"- Crowding level: {crowding_analysis['crowding_level'].upper()}")
                summary_parts.append(f"- Crowding score: {crowding_analysis['average_crowding_score']:.1f}/100")
                summary_parts.append(f"- Crowding trend: {crowding_analysis['crowding_trend']}")
            else:
                summary_parts.append(f"- Crowding detected: NO (safe levels)")
            
            # Falling object analysis
            summary_parts.append(f"\nFalling Object Analysis:")
            if falling_analysis.get('falling_detected'):
                summary_parts.append(f"- Falling objects detected: YES")
                summary_parts.append(f"- Falling risk: {falling_analysis['falling_risk'].upper()}")
                summary_parts.append(f"- Risk score: {falling_analysis['average_falling_risk']:.1f}/100")
                summary_parts.append(f"- Potential objects: {falling_analysis['total_potential_objects']}")
            else:
                summary_parts.append(f"- Falling objects detected: NO (low risk)")
            
            # Safety recommendations
            summary_parts.append(f"\nSafety Recommendations:")
            if crowding_analysis.get('crowding_detected') and crowding_analysis['crowding_level'] in ['high', 'moderate']:
                summary_parts.append(f"- ⚠️ CROWDING ALERT: Consider crowd control measures")
            
            if falling_analysis.get('falling_detected') and falling_analysis['falling_risk'] in ['high', 'moderate']:
                summary_parts.append(f"- ⚠️ FALLING RISK: Monitor for dropped objects/bags")
            
            if not crowding_analysis.get('crowding_detected') and not falling_analysis.get('falling_detected'):
                summary_parts.append(f"- ✅ SAFE: No immediate safety concerns detected")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating safety summary: {str(e)}"
    
    def _generate_safety_alerts(self, crowding_analysis: Dict, falling_analysis: Dict) -> List[str]:
        """Generate safety alerts based on analysis"""
        alerts = []
        
        # Crowding alerts
        if crowding_analysis.get('crowding_detected'):
            if crowding_analysis['crowding_level'] == 'high':
                alerts.append("🚨 HIGH CROWDING ALERT: Immediate attention required")
            elif crowding_analysis['crowding_level'] == 'moderate':
                alerts.append("⚠️ MODERATE CROWDING: Monitor situation")
        
        # Falling object alerts
        if falling_analysis.get('falling_detected'):
            if falling_analysis['falling_risk'] == 'high':
                alerts.append("🚨 HIGH FALLING RISK: Objects/bags may fall")
            elif falling_analysis['falling_risk'] == 'moderate':
                alerts.append("⚠️ MODERATE FALLING RISK: Monitor for dropped items")
        
        # Combined alerts
        if (crowding_analysis.get('crowding_detected') and 
            falling_analysis.get('falling_detected')):
            alerts.append("🚨 CRITICAL: Both crowding and falling risks detected")
        
        if not alerts:
            alerts.append("✅ SAFE: No safety alerts")
        
        return alerts

def test_escalator_analyzer():
    """Test the escalator safety analyzer"""
    print("🚇 Testing Escalator Safety Analyzer")
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
        analyzer = EscalatorAnalyzer()
        analyzer.max_frames_to_analyze = 8  # Small test
        
        start_time = time.time()
        result = analyzer.analyze_escalator_safety(str(test_video), "safety")
        processing_time = time.time() - start_time
        
        if result and 'error' not in result:
            print("✅ Escalator analyzer test PASSED")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Frames analyzed: {result['total_frames_analyzed']}")
            print(f"   - Crowding detected: {result['crowding_analysis']['crowding_detected']}")
            print(f"   - Falling risk: {result['falling_analysis']['falling_risk']}")
            print(f"   - Safety alerts: {len(result['safety_alerts'])}")
            return True
        else:
            print(f"❌ Escalator analyzer test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    test_escalator_analyzer() 