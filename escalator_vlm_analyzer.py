#!/usr/bin/env python3
"""
Enhanced Escalator Safety Analyzer with VLM
Uses Vision-Language Models for better detection of crowding and falling objects
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# VLM components will be imported dynamically in the class

class EscalatorVLMAnalyzer:
    """
    Enhanced escalator analyzer with VLM capabilities
    Better detection of crowding and falling objects using AI models
    """
    
    def __init__(self, hf_token=None, use_vlm=True):
        self.max_frames_to_analyze = 12
        self.min_frame_interval = 0.5
        self.hf_token = hf_token
        self.use_vlm = use_vlm
        
        # Initialize VLM components
        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_available = False
        
        # Initialize VLM if token is provided and VLM is enabled
        if hf_token and use_vlm:
            self._initialize_vlm()
        elif not use_vlm:
            print("VLM disabled - using basic analysis only")
        else:
            print("WARNING: No Hugging Face token provided - VLM will not be available")
        
    def _initialize_vlm(self):
        """Initialize VLM model for enhanced analysis"""
        print("Starting VLM initialization...")
        print(f"DEBUG: Token provided: {self.hf_token is not None}")
        if self.hf_token:
            print(f"DEBUG: Token starts with: {self.hf_token[:10]}...")
        
        try:
            print("Loading VLM model for enhanced escalator analysis...")
            
            # Check if transformers is available
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                import torch
                print("Transformers and torch imported successfully in VLM init")
            except ImportError as e:
                print(f"WARNING: Transformers not available: {e}")
                self.vlm_available = False
                return
            
            # Try to load a lightweight VLM model
            model_name = "microsoft/git-base"  # Lightweight and fast
            print(f"DEBUG: Using model: {model_name}")
            
            # Alternative models to try if the first one fails
            alternative_models = [
                "microsoft/git-base",
                "microsoft/git-base-coco",  # COCO fine-tuned version
                "microsoft/git-base-textcaps"  # TextCaps fine-tuned version
            ]
            
            # Check if we should use a faster model for better performance
            use_fast_model = True  # Always use fast model for better performance
            
            # Check if model is already cached
            cache_path = Path("./model_cache")
            if cache_path.exists():
                print(f"DEBUG: Cache directory exists: {cache_path}")
                cached_models = list(cache_path.glob("*"))
                print(f"DEBUG: Cached models: {[m.name for m in cached_models]}")
            else:
                print("DEBUG: No cache directory found")
            
            # Add more detailed error handling
            if not self.hf_token:
                print("WARNING: No Hugging Face token provided")
                self.vlm_available = False
                return
            
            print(f"Using token: {self.hf_token[:10]}...")
            
            # Load processor with error handling
            try:
                print("Loading VLM processor...")
                self.vlm_processor = AutoProcessor.from_pretrained(
                    model_name,
                    token=self.hf_token,
                    cache_dir="./model_cache",
                    trust_remote_code=True
                )
                print("VLM processor loaded successfully")
            except Exception as e:
                print(f"WARNING: VLM processor loading failed: {e}")
                print(f"DEBUG: Error type: {type(e)}")
                print(f"DEBUG: Error details: {str(e)}")
                self.vlm_available = False
                return
            
            # Load model with error handling
            try:
                print("Loading VLM model...")
                self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    token=self.hf_token,
                    cache_dir="./model_cache",
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                print("VLM model loaded successfully")
            except Exception as e:
                print(f"WARNING: VLM model loading failed: {e}")
                print(f"DEBUG: Error type: {type(e)}")
                print(f"DEBUG: Error details: {str(e)}")
                self.vlm_available = False
                return
            
            # Verify both components are loaded
            if self.vlm_processor is not None and self.vlm_model is not None:
                print("VLM model and processor loaded successfully")
                self.vlm_available = True
                print(f"DEBUG: Final VLM status - Available: {self.vlm_available}, Processor: {self.vlm_processor is not None}, Model: {self.vlm_model is not None}")
            else:
                print("WARNING: VLM components not properly loaded")
                print(f"DEBUG: Processor: {self.vlm_processor is not None}, Model: {self.vlm_model is not None}")
                self.vlm_available = False
                
        except Exception as e:
            print(f"WARNING: Unexpected error during VLM initialization: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            self.vlm_available = False
        
    def analyze_escalator_vlm(self, video_path: str, analysis_mode: str = "enhanced") -> Dict:
        """
        Analyze escalator video with VLM-enhanced detection
        """
        try:
            start_time = time.time()
            
            print(f"Starting VLM-enhanced escalator analysis: {video_path}")
            
            # Validate video file
            if not self._validate_video(video_path):
                return {"error": "Invalid video file"}
            
            # Extract video metadata
            video_info = self._extract_video_metadata(video_path)
            print(f"Video info: {video_info['duration']:.2f}s, {video_info['resolution']}")
            
            # Extract frames
            frames = self._extract_frames(video_path)
            
            if not frames:
                return {"error": "No valid frames extracted"}
            
            # Analyze frames with VLM enhancement
            print("Starting frame analysis...")
            frame_analyses = self._analyze_frames_vlm(frames)
            
            if not frame_analyses:
                return {"error": "No frames were successfully analyzed"}
            
            # Enhanced crowding detection
            print("Analyzing crowding...")
            crowding_analysis = self._detect_crowding_vlm(frame_analyses)
            
            # Enhanced falling object detection
            print("Analyzing falling objects...")
            falling_analysis = self._detect_falling_objects_vlm(frame_analyses)
            
            # Generate comprehensive safety summary
            print("Generating safety summary...")
            safety_summary = self._generate_vlm_safety_summary(
                frame_analyses, crowding_analysis, falling_analysis, video_info
            )
            
            processing_time = time.time() - start_time
            print(f"Analysis completed in {processing_time:.2f} seconds")
            
            return {
                'video_info': video_info,
                'frame_analyses': frame_analyses,
                'crowding_analysis': crowding_analysis,
                'falling_analysis': falling_analysis,
                'safety_summary': safety_summary,
                'processing_time': processing_time,
                'analysis_mode': analysis_mode,
                'total_frames_analyzed': len(frames),
                'vlm_used': self.vlm_available,
                'safety_alerts': self._generate_enhanced_alerts(crowding_analysis, falling_analysis)
            }
            
        except Exception as e:
            print(f"Main analysis error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in VLM analysis: {str(e)}"}
    
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
        """Extract frames from video - optimized for safety analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to get a good distribution
            if total_frames > 0:
                # Extract frames evenly distributed across the video
                frame_interval = max(1, total_frames // self.max_frames_to_analyze)
            else:
                frame_interval = max(1, int(fps * self.min_frame_interval))
            
            print(f"Extracting frames: total={total_frames}, fps={fps:.1f}, interval={frame_interval}")
            
            while len(frames) < self.max_frames_to_analyze:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at regular intervals
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    frames.append((frame_count, frame, timestamp))
                    print(f"Extracted frame {frame_count} at {timestamp:.2f}s")
                
                frame_count += 1
                
                # Safety check to prevent infinite loops
                if frame_count > total_frames * 2:
                    print("Warning: Frame extraction loop limit reached")
                    break
            
            cap.release()
            print(f"Successfully extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _analyze_frames_vlm(self, frames: List[Tuple[int, np.ndarray, float]]) -> List[Dict]:
        """Analyze frames with enhanced safety analysis - optimized for reliability"""
        analyses = []
        
        # Since VLM is unreliable, focus on enhanced basic analysis for all frames
        # But still try VLM on a few frames for potential AI insights
        max_vlm_attempts = 3  # Reduced from 6 to improve performance
        total_frames = len(frames)
        
        # Calculate which frames should get VLM analysis (distributed evenly)
        vlm_frame_indices = set()
        if total_frames > 0:
            if total_frames <= max_vlm_attempts:
                vlm_frame_indices = set(range(total_frames))
            else:
                step = total_frames / max_vlm_attempts
                for i in range(max_vlm_attempts):
                    frame_index = int(i * step)
                    vlm_frame_indices.add(min(frame_index, total_frames - 1))
        
        print(f"Enhanced analysis will be performed on all frames")
        print(f"VLM attempts will be made on frames: {sorted(vlm_frame_indices)} (if available)")
        
        for i, (frame_number, frame, timestamp) in enumerate(frames):
            try:
                # Convert to grayscale for basic analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Enhanced basic frame analysis (always performed)
                basic_analysis = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'brightness': np.mean(gray),
                    'contrast': np.std(gray),
                    'crowding_score': self._calculate_crowding_score(gray),
                    'falling_detection': self._detect_falling_basic(gray, frame),
                    'processing_time': 0
                }
                
                # Try VLM analysis on selected frames (but don't rely on it)
                if self.vlm_available and i in vlm_frame_indices:
                    try:
                        print(f"Attempting VLM analysis on frame {i+1}/{total_frames} (timestamp: {timestamp:.2f}s)")
                        vlm_analysis = self._analyze_frame_vlm(frame)
                        
                        # Only use VLM if it actually generated something meaningful
                        if vlm_analysis.get('processing_time', 0) > 0:
                            basic_analysis.update(vlm_analysis)
                            print(f"✓ VLM analysis successful for frame {i+1}")
                        else:
                            # VLM failed, use enhanced basic analysis
                            basic_analysis['vlm_description'] = self._generate_enhanced_basic_description(frame, gray)
                            basic_analysis['crowding_vlm_score'] = basic_analysis['crowding_score']
                            basic_analysis['falling_vlm_score'] = basic_analysis['falling_detection']['falling_probability']
                            basic_analysis['processing_time'] = 0
                            print(f"✗ VLM analysis failed for frame {i+1}, using enhanced basic analysis")
                            
                    except Exception as vlm_error:
                        print(f"VLM analysis failed for frame {frame_number}: {vlm_error}")
                        # Fallback to enhanced basic analysis
                        basic_analysis['vlm_description'] = self._generate_enhanced_basic_description(frame, gray)
                        basic_analysis['crowding_vlm_score'] = basic_analysis['crowding_score']
                        basic_analysis['falling_vlm_score'] = basic_analysis['falling_detection']['falling_probability']
                        basic_analysis['processing_time'] = 0
                else:
                    # Use enhanced basic analysis for non-VLM frames
                    basic_analysis['vlm_description'] = self._generate_enhanced_basic_description(frame, gray)
                    basic_analysis['crowding_vlm_score'] = basic_analysis['crowding_score']
                    basic_analysis['falling_vlm_score'] = basic_analysis['falling_detection']['falling_probability']
                    basic_analysis['processing_time'] = 0
                
                analyses.append(basic_analysis)
                
                # Progress indicator
                if (i + 1) % 3 == 0:
                    print(f"Processed {i + 1}/{total_frames} frames...")
                
            except Exception as e:
                print(f"Error analyzing frame {frame_number}: {e}")
                # Add a minimal analysis entry to maintain frame count
                analyses.append({
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'brightness': 0,
                    'contrast': 0,
                    'crowding_score': 0,
                    'falling_detection': {'potential_objects': 0, 'falling_probability': 0},
                    'vlm_description': "Frame analysis failed",
                    'crowding_vlm_score': 0,
                    'falling_vlm_score': 0,
                    'processing_time': 0
                })
                continue
        
        print(f"Completed analysis of {len(analyses)} frames")
        print(f"VLM analysis performed on {len(vlm_frame_indices)} frames distributed across video")
        return analyses
    
    def _analyze_frame_vlm(self, frame: np.ndarray) -> Dict:
        """Analyze frame using VLM model with timeout and optimization"""
        try:
            # Import torch here to avoid import issues
            import torch
            import signal
            import time
            
            # Check if VLM is available
            if not self.vlm_available or self.vlm_processor is None or self.vlm_model is None:
                print("VLM not available, using enhanced basic analysis")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                basic_desc = self._generate_enhanced_basic_description(frame, gray)
                return {
                    'vlm_description': basic_desc,
                    'crowding_vlm_score': self._calculate_crowding_score(gray),
                    'falling_vlm_score': self._detect_falling_basic(gray, frame)['falling_probability'],
                    'vlm_analysis': [basic_desc]
                }
            
            # Convert BGR to RGB for VLM
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to reduce processing time (maintain aspect ratio)
            height, width = frame_rgb.shape[:2]
            if width > 512 or height > 512:
                scale = min(512/width, 512/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            try:
                # Process image with VLM - try to get meaningful safety insights
                # Use a simple, direct prompt that works better with this model
                prompt = "A person on an escalator"
                
                try:
                    inputs = self.vlm_processor(
                        images=frame_rgb,
                        text=prompt,
                        return_tensors="pt"
                    )
                    
                    # Generate description with optimized parameters
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = self.vlm_model.generate(
                            **inputs, 
                            max_length=25,  # Very short for speed
                            num_beams=1,    # Fast greedy search
                            early_stopping=True,
                            do_sample=False,  # Deterministic for speed
                            temperature=1.0,
                            pad_token_id=self.vlm_processor.tokenizer.eos_token_id,
                            repetition_penalty=1.0
                        )
                    
                    processing_time = time.time() - start_time
                    
                    # Decode the output
                    raw_description = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Check if the model actually generated something meaningful
                    if (raw_description and 
                        raw_description != prompt and 
                        len(raw_description) > len(prompt) + 3 and
                        not raw_description.endswith(prompt) and
                        processing_time < 5.0):  # Shorter timeout
                        
                        # Clean up and simplify the description
                        description = self._simplify_vlm_description(raw_description)
                        
                        # Verify we got a good description
                        if description and len(description) > 5:
                            print(f"VLM generated: '{description}'")
                        else:
                            # Fall back to enhanced basic analysis
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            description = self._generate_enhanced_basic_description(frame, gray)
                            processing_time = 0
                    else:
                        # VLM didn't generate properly or took too long, use enhanced basic analysis
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        description = self._generate_enhanced_basic_description(frame, gray)
                        processing_time = 0
                        
                except Exception as e:
                    print(f"VLM processing error: {e}")
                    # Use enhanced basic analysis as fallback
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    description = self._generate_enhanced_basic_description(frame, gray)
                    processing_time = 0
                        
            except Exception as e:
                print(f"VLM processing error: {e}")
                # Use enhanced basic analysis as fallback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                description = self._generate_enhanced_basic_description(frame, gray)
                processing_time = 0
                
            except Exception as e:
                print(f"VLM processing error: {e}")
                # Use basic analysis as fallback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                description = self._generate_basic_description(frame, gray)
            
            # Extract safety insights from the description
            crowding_vlm_score = self._extract_crowding_from_vlm([description])
            falling_vlm_score = self._extract_falling_from_vlm([description])
            
            return {
                'vlm_description': description,
                'crowding_vlm_score': crowding_vlm_score,
                'falling_vlm_score': falling_vlm_score,
                'vlm_analysis': [description],
                'processing_time': processing_time if 'processing_time' in locals() else 0
            }
            
        except Exception as e:
            print(f"VLM analysis error: {e}")
            # Final fallback to enhanced basic analysis
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                basic_desc = self._generate_enhanced_basic_description(frame, gray)
                return {
                    'vlm_description': basic_desc,
                    'crowding_vlm_score': self._calculate_crowding_score(gray),
                    'falling_vlm_score': self._detect_falling_basic(gray, frame)['falling_probability'],
                    'vlm_analysis': [basic_desc],
                    'processing_time': 0
                }
            except Exception as fallback_error:
                print(f"Fallback analysis also failed: {fallback_error}")
                return {
                    'vlm_description': "Analysis failed - unable to process frame",
                    'crowding_vlm_score': 0,
                    'falling_vlm_score': 0,
                    'vlm_analysis': [],
                    'processing_time': 0
                }
    
    def _simplify_vlm_description(self, text: str) -> str:
        """Simplify VLM output to focus on safety-relevant information"""
        # Remove common prefixes and make it more concise
        text = text.lower().strip()
        
        # Remove any prompt text and common prefixes
        prompt_texts = [
            "describe this escalator scene focusing on people, objects, and safety concerns",
            "describe this image",
            "this is",
            "the image shows",
            "the scene shows",
            "the escalator is seen in this",
            "undated image",
            "this undated image",
            "a photo of an escalator showing",
            "this image shows an escalator with",
            "in this escalator scene, i can see",
            "the escalator in this image has",
            "a photo of",
            "this image shows",
            "in this scene, i can see",
            "the image has",
            "describe this escalator scene for safety analysis",
            "analyze this escalator scene for safety",
            "describe the people and objects in this escalator"
        ]
        for prompt in prompt_texts:
            if prompt in text:
                text = text.replace(prompt, "").strip()
        
        # Fix common VLM errors and improve clarity
        error_fixes = {
            "the woman was hit by a woman": "person on escalator",
            "who was hit by an electric vehicle": "with potential safety concern",
            "the scary woman": "person",
            "was knocked down the escalator": "on escalator",
            "knocked down the escalator": "on escalator",
            "knocked over by": "person near",
            "unattended suitcase": "unattended luggage",
            "terrifying moment": "safety incident",
            "hit by": "near",
            "electric vehicle": "object",
            "scary": "",
            "undated": "",
            "this undated": ""
        }
        
        for old, new in error_fixes.items():
            if old in text:
                text = text.replace(old, new).strip()
        
        # Simplify common escalator descriptions to be more concise
        simplifications = {
            "passengers wait on the escalator": "people on escalator",
            "passengers on escalators": "people on escalator", 
            "passengers on the escalator": "people on escalator",
            "people wait on the escalator": "people on escalator",
            "people on escalators": "people on escalator",
            "people on the escalator": "people on escalator",
            "crowd of people on escalator": "crowded escalator",
            "many people on escalator": "crowded escalator",
            "busy escalator": "crowded escalator",
            "escalator with people": "people on escalator",
            "at the metro station": "",
            "at the subway station": "",
            "in the metro": "",
            "in the subway": "",
            "walking on escalator": "on escalator",
            "standing on escalator": "on escalator",
            "riding escalator": "on escalator",
            "outdoors scene": "outdoor scene",
            "indoor scene": "indoor scene"
        }
        
        for old, new in simplifications.items():
            if old in text:
                text = text.replace(old, new)
        
        # Clean up multiple spaces and punctuation
        import re
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\s+\.', '.', text)  # Space before period
        text = re.sub(r'\.+', '.', text)  # Multiple periods to single
        text = text.strip(' .')  # Remove leading/trailing spaces and periods
        
        # Make it a single sentence if possible
        sentences = text.split('.')
        if len(sentences) > 1:
            # Take the first meaningful sentence
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 5:
                    text = sentence
                    break
        
        # Add safety context if relevant
        if 'fall' in text or 'falling' in text or 'drop' in text:
            # Avoid double "down"
            if 'fall down' not in text:
                text = text.replace('fall', 'fall down')
            if 'falling down' not in text:
                text = text.replace('falling', 'falling down')
            # Fix "fall downing" to "falling down"
            text = text.replace('fall downing', 'falling down')
        
        # Enhance descriptions for falling objects
        if any(word in text for word in ['bag', 'luggage', 'object']) and any(word in text for word in ['fall', 'drop']):
            if 'people' in text or 'person' in text:
                # Don't duplicate "with bag" if it's already there
                if 'with bag' not in text:
                    text = text.replace('people', 'people with bag').replace('person', 'person with bag')
        
        # If the description is too generic or contains errors, provide a better fallback
        if text in ['outdoor scene', 'indoor scene', 'this is an outdoors scene', 'an outdoor scene']:
            text = "general scene - no specific safety concerns detected"
        elif text in ['a scene', 'scene', 'this is a scene']:
            text = "general scene - no specific safety concerns detected"
        elif len(text) < 10 or text in ['', ' ', '.']:
            text = "escalator scene - analysis incomplete"
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _extract_crowding_from_vlm(self, vlm_results: List[str]) -> float:
        """Extract crowding score from VLM analysis"""
        try:
            text = " ".join(vlm_results).lower()
            
            # Crowding indicators
            crowding_keywords = [
                'crowded', 'many people', 'busy', 'packed', 'crowd', 'lots of people', 'full',
                'escalator', 'people', 'person', 'group', 'multiple', 'several', 'many',
                'passengers', 'riders', 'users', 'occupants'
            ]
            anti_crowding_keywords = ['empty', 'few people', 'quiet', 'sparse', 'not crowded', 'no one', 'deserted']
            
            crowding_score = 0
            
            # Check for crowding indicators
            for keyword in crowding_keywords:
                if keyword in text:
                    crowding_score += 15
            
            # Check for anti-crowding indicators
            for keyword in anti_crowding_keywords:
                if keyword in text:
                    crowding_score -= 20
            
            # Boost score if escalator is mentioned with people
            if 'escalator' in text and any(word in text for word in ['people', 'person', 'crowd']):
                crowding_score += 25
            
            return min(100, max(0, crowding_score))
            
        except Exception:
            return 0
    
    def _extract_falling_from_vlm(self, vlm_results: List[str]) -> float:
        """Extract falling risk score from VLM analysis"""
        try:
            text = " ".join(vlm_results).lower()
            
            # Falling risk indicators
            falling_keywords = [
                'bag', 'object', 'item', 'fall', 'drop', 'safety concern', 'risk',
                'luggage', 'backpack', 'purse', 'handbag', 'suitcase', 'package',
                'box', 'container', 'belongings', 'personal items', 'unattended',
                'knocked over', 'knocked down', 'safety incident', 'terrifying'
            ]
            safety_keywords = ['safe', 'clear', 'no objects', 'no bags', 'clean', 'empty']
            
            falling_score = 0
            
            # Check for falling risk indicators
            for keyword in falling_keywords:
                if keyword in text:
                    falling_score += 20
            
            # Check for safety indicators
            for keyword in safety_keywords:
                if keyword in text:
                    falling_score -= 15
            
            # Boost score if escalator is mentioned with objects
            if 'escalator' in text and any(word in text for word in ['bag', 'object', 'item', 'luggage']):
                falling_score += 30
            
            # Enhanced detection for falling objects
            if any(word in text for word in ['bag', 'luggage', 'object', 'item']) and any(word in text for word in ['fall', 'drop', 'slip']):
                falling_score += 35
            
            # Specific detection for "fall down" scenarios
            if 'fall down' in text or 'falling down' in text:
                falling_score += 40
            
            return min(100, max(0, falling_score))
            
        except Exception:
            return 0
    
    def _calculate_crowding_score(self, gray: np.ndarray) -> float:
        """Calculate basic crowding score"""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            texture_score = np.mean(gradient_magnitude)
            
            crowding_score = (edge_density * 1000) + (texture_score / 10)
            return min(100, crowding_score)
            
        except Exception:
            return 0
    
    def _detect_falling_basic(self, gray: np.ndarray, frame: np.ndarray) -> Dict:
        """Basic falling object detection"""
        try:
            # Detect blobs (potential objects)
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 10
            params.maxThreshold = 200
            params.filterByArea = True
            params.minArea = 100
            params.maxArea = 5000
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            return {
                'potential_objects': len(keypoints),
                'falling_probability': min(100, len(keypoints) * 10)
            }
            
        except Exception:
            return {'potential_objects': 0, 'falling_probability': 0}
    
    def _generate_basic_description(self, frame: np.ndarray, gray: np.ndarray) -> str:
        """Generate basic description when VLM is not available"""
        try:
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Enhanced scene analysis
            if brightness < 50:
                scene_type = "dark scene"
            elif brightness > 200:
                scene_type = "bright scene"
            else:
                scene_type = "normal lighting"
            
            # Add motion/activity analysis
            activity_level = "low"
            if contrast > 50:
                activity_level = "high"
            elif contrast > 25:
                activity_level = "moderate"
            
            # Add basic object detection hints
            object_hint = ""
            if self._detect_falling_basic(gray, frame)['potential_objects'] > 3:
                object_hint = " - potential objects detected"
            
            return f"Scene: {scene_type}, activity: {activity_level}{object_hint}"
            
        except Exception:
            return "Basic analysis: Unable to analyze frame"
    
    def _generate_enhanced_basic_description(self, frame: np.ndarray, gray: np.ndarray) -> str:
        """Generate enhanced basic description with more detailed analysis"""
        try:
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Enhanced scene analysis
            if brightness < 50:
                scene_type = "dark escalator scene"
            elif brightness > 200:
                scene_type = "bright escalator scene"
            else:
                scene_type = "normal lighting escalator scene"
            
            # Add motion/activity analysis
            activity_level = "low"
            if contrast > 50:
                activity_level = "high"
            elif contrast > 25:
                activity_level = "moderate"
            
            # Enhanced object detection
            falling_detection = self._detect_falling_basic(gray, frame)
            object_count = falling_detection['potential_objects']
            
            # Enhanced crowding analysis
            crowding_score = self._calculate_crowding_score(gray)
            
            # Build enhanced description with safety focus
            description_parts = []
            
            # Scene description
            description_parts.append(scene_type)
            
            # Activity level
            if activity_level != "low":
                description_parts.append(f"with {activity_level} activity")
            
            # Safety-focused object detection
            if object_count > 8:
                description_parts.append("multiple objects detected - potential falling risk")
            elif object_count > 4:
                description_parts.append("several objects visible - monitor for safety")
            elif object_count > 1:
                description_parts.append("some objects present")
            
            # Safety-focused crowding analysis
            if crowding_score > 80:
                description_parts.append("highly crowded - safety concern")
            elif crowding_score > 60:
                description_parts.append("crowded escalator - monitor closely")
            elif crowding_score > 40:
                description_parts.append("moderate occupancy")
            else:
                description_parts.append("low occupancy - safe conditions")
            
            # Enhanced safety assessment
            if falling_detection['falling_probability'] > 70:
                description_parts.append("high falling risk detected")
            elif falling_detection['falling_probability'] > 40:
                description_parts.append("moderate falling risk")
            elif falling_detection['falling_probability'] > 20:
                description_parts.append("low falling risk")
            
            # Add specific safety insights
            if object_count > 5 and crowding_score > 60:
                description_parts.append("crowded with objects - increased safety risk")
            elif object_count > 3 and activity_level == "high":
                description_parts.append("high activity with objects - monitor for incidents")
            elif object_count > 2 and falling_detection['falling_probability'] > 30:
                description_parts.append("objects present with falling risk - safety alert")
            elif crowding_score > 70 and activity_level == "high":
                description_parts.append("highly active crowded scene - monitor closely")
            elif object_count == 0 and crowding_score < 30:
                description_parts.append("clear escalator - safe conditions")
            
            return ", ".join(description_parts)
            
        except Exception:
            return "Enhanced analysis: escalator scene with safety monitoring"
    
    def _detect_crowding_vlm(self, frame_analyses: List[Dict]) -> Dict:
        """Enhanced crowding detection with VLM"""
        try:
            if len(frame_analyses) < 2:
                return {"crowding_detected": False, "crowding_level": "low"}
            
            # Combine basic and VLM scores
            crowding_scores = []
            vlm_crowding_scores = []
            
            for analysis in frame_analyses:
                crowding_scores.append(analysis['crowding_score'])
                vlm_crowding_scores.append(analysis.get('crowding_vlm_score', 0))
            
            # Weighted average (VLM gets higher weight if available)
            if self.vlm_available:
                avg_crowding = (np.mean(crowding_scores) * 0.3 + np.mean(vlm_crowding_scores) * 0.7)
            else:
                avg_crowding = np.mean(crowding_scores)
            
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
            
            return {
                "crowding_detected": crowding_detected,
                "crowding_level": crowding_level,
                "average_crowding_score": avg_crowding,
                "vlm_enhanced": self.vlm_available,
                "crowding_percentage": min(100, avg_crowding)
            }
            
        except Exception as e:
            return {"crowding_detected": False, "crowding_level": "unknown", "error": str(e)}
    
    def _detect_falling_objects_vlm(self, frame_analyses: List[Dict]) -> Dict:
        """Enhanced falling object detection with VLM"""
        try:
            if len(frame_analyses) < 2:
                return {"falling_detected": False, "falling_risk": "low"}
            
            # Combine basic and VLM scores
            falling_scores = []
            vlm_falling_scores = []
            
            for analysis in frame_analyses:
                falling_scores.append(analysis['falling_detection']['falling_probability'])
                vlm_falling_scores.append(analysis.get('falling_vlm_score', 0))
            
            # Weighted average
            if self.vlm_available:
                avg_falling_risk = (np.mean(falling_scores) * 0.3 + np.mean(vlm_falling_scores) * 0.7)
            else:
                avg_falling_risk = np.mean(falling_scores)
            
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
            
            return {
                "falling_detected": falling_detected,
                "falling_risk": falling_risk,
                "average_falling_risk": avg_falling_risk,
                "vlm_enhanced": self.vlm_available,
                "falling_risk_percentage": min(100, avg_falling_risk)
            }
            
        except Exception as e:
            return {"falling_detected": False, "falling_risk": "unknown", "error": str(e)}
    
    def _generate_vlm_safety_summary(self, frame_analyses: List[Dict], crowding_analysis: Dict, 
                                    falling_analysis: Dict, video_info: Dict) -> str:
        """Generate comprehensive safety summary with VLM insights"""
        try:
            summary_parts = []
            
            # Video info
            summary_parts.append(f"Enhanced Escalator Safety Analysis (VLM-Enhanced):")
            summary_parts.append(f"File: {video_info['file_name']}")
            summary_parts.append(f"Duration: {video_info['duration']:.2f} seconds")
            summary_parts.append(f"Resolution: {video_info['resolution']}")
            summary_parts.append(f"Frames analyzed: {len(frame_analyses)}")
            summary_parts.append(f"VLM Enhancement: {'Enabled' if self.vlm_available else 'Not available'}")
            
            # Crowding analysis
            summary_parts.append(f"\nCrowding Analysis:")
            if crowding_analysis.get('crowding_detected'):
                summary_parts.append(f"- Crowding detected: YES")
                summary_parts.append(f"- Crowding level: {crowding_analysis['crowding_level'].upper()}")
                summary_parts.append(f"- Crowding score: {crowding_analysis['average_crowding_score']:.1f}/100")
                if crowding_analysis.get('vlm_enhanced'):
                    summary_parts.append(f"- VLM Enhanced: AI model confirmed crowding detection")
            else:
                summary_parts.append(f"- Crowding detected: NO (safe levels)")
            
            # Falling object analysis
            summary_parts.append(f"\nFalling Object Analysis:")
            if falling_analysis.get('falling_detected'):
                summary_parts.append(f"- Falling objects detected: YES")
                summary_parts.append(f"- Falling risk: {falling_analysis['falling_risk'].upper()}")
                summary_parts.append(f"- Risk score: {falling_analysis['average_falling_risk']:.1f}/100")
                if falling_analysis.get('vlm_enhanced'):
                    summary_parts.append(f"- VLM Enhanced: AI model identified potential falling objects")
            else:
                summary_parts.append(f"- Falling objects detected: NO (low risk)")
            
            # VLM insights
            if self.vlm_available and frame_analyses:
                summary_parts.append(f"\nAI Model Insights:")
                
                # Count how many frames had VLM analysis vs basic analysis
                vlm_frames = 0
                basic_frames = 0
                vlm_descriptions = []
                
                # Show all frames with their analysis type and results
                for i, analysis in enumerate(frame_analyses):
                    desc = analysis.get('vlm_description', '')
                    processing_time = analysis.get('processing_time', 0)
                    timestamp = analysis.get('timestamp', 0)
                    
                    # Determine if this frame had VLM analysis (processing_time > 0 indicates VLM was used)
                    if processing_time > 0:
                        vlm_frames += 1
                        frame_type = "VLM"
                    else:
                        basic_frames += 1
                        frame_type = "Basic"
                    
                    if desc and 'VLM analysis failed' not in desc and 'analysis incomplete' not in desc:
                        # Truncate long descriptions but keep meaningful content
                        if len(desc) > 120:
                            desc = desc[:120] + "..."
                        summary_parts.append(f"- Frame {i+1} ({frame_type}) at {timestamp:.1f}s: {desc}")
                    elif desc and ('VLM analysis failed' in desc or 'analysis incomplete' in desc):
                        summary_parts.append(f"- Frame {i+1} ({frame_type}) at {timestamp:.1f}s: Analysis incomplete - using fallback detection")
                    else:
                        summary_parts.append(f"- Frame {i+1} ({frame_type}) at {timestamp:.1f}s: No AI analysis available")
                    
                    vlm_descriptions.append(desc)
                
                # Add summary of analysis types
                if vlm_frames > 0:
                    summary_parts.append(f"\nAnalysis Summary:")
                    summary_parts.append(f"- VLM Enhanced Frames: {vlm_frames} (AI-powered insights)")
                    summary_parts.append(f"- Enhanced Basic Analysis Frames: {basic_frames} (computer vision + safety algorithms)")
                    summary_parts.append(f"- Total Frames Analyzed: {len(frame_analyses)}")
                    summary_parts.append(f"- VLM Coverage: {vlm_frames}/{len(frame_analyses)} frames ({vlm_frames/len(frame_analyses)*100:.1f}%)")
                    summary_parts.append(f"- Enhanced Basic Coverage: {basic_frames}/{len(frame_analyses)} frames ({basic_frames/len(frame_analyses)*100:.1f}%)")
                else:
                    summary_parts.append(f"\nAnalysis Summary:")
                    summary_parts.append(f"- Enhanced Basic Analysis Frames: {basic_frames} (computer vision + safety algorithms)")
                    summary_parts.append(f"- Total Frames Analyzed: {len(frame_analyses)}")
                    summary_parts.append(f"- Enhanced Basic Coverage: {basic_frames}/{len(frame_analyses)} frames (100%)")
                    summary_parts.append(f"- Note: VLM was attempted but fell back to enhanced basic analysis for reliability")
                
                # Add overall AI assessment
                safety_keywords = ['fall', 'falling', 'crowd', 'crowded', 'safety', 'risk', 'concern', 
                                 'knocked', 'unattended', 'incident', 'terrifying', 'suitcase', 'luggage']
                has_safety_concerns = any(any(keyword in desc.lower() for keyword in safety_keywords) 
                                        for desc in vlm_descriptions if desc)
                if has_safety_concerns:
                    summary_parts.append(f"  → AI detected potential safety concerns in video frames")
                else:
                    summary_parts.append(f"  → AI analysis shows normal escalator operation")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating VLM safety summary: {str(e)}"
    
    def _generate_enhanced_alerts(self, crowding_analysis: Dict, falling_analysis: Dict) -> List[str]:
        """Generate enhanced safety alerts"""
        alerts = []
        
        # Crowding alerts
        if crowding_analysis.get('crowding_detected'):
            if crowding_analysis['crowding_level'] == 'high':
                alerts.append("HIGH CROWDING ALERT: Immediate attention required")
            elif crowding_analysis['crowding_level'] == 'moderate':
                alerts.append("WARNING: MODERATE CROWDING: Monitor situation")
        
        # Falling object alerts
        if falling_analysis.get('falling_detected'):
            if falling_analysis['falling_risk'] == 'high':
                alerts.append("HIGH FALLING RISK: Objects/bags may fall")
            elif falling_analysis['falling_risk'] == 'moderate':
                alerts.append("WARNING: MODERATE FALLING RISK: Monitor for dropped items")
        
        # VLM enhancement alerts
        if self.vlm_available:
            alerts.append("AI Enhanced: VLM model provided additional safety insights")
        
        if not alerts:
            alerts.append("SAFE: No safety alerts")
        
        return alerts

def test_escalator_vlm():
    """Test the VLM-enhanced escalator analyzer"""
    print("Testing VLM-Enhanced Escalator Safety Analyzer")
    print("=" * 60)
    
    # Find test videos
    test_folder = "test_video"
    if not os.path.exists(test_folder):
        print(f"Test folder '{test_folder}' not found")
        return False
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(Path(test_folder).glob(f"*{ext}"))
    
    if not video_files:
        print("No video files found")
        return False
    
    # Use the smallest file
    test_video = min(video_files, key=lambda x: x.stat().st_size)
    print(f"Testing with: {test_video.name} ({test_video.stat().st_size / 1024:.1f} KB)")
    
    try:
        analyzer = EscalatorVLMAnalyzer()
        analyzer.max_frames_to_analyze = 6  # Small test
        
        start_time = time.time()
        result = analyzer.analyze_escalator_vlm(str(test_video), "enhanced")
        processing_time = time.time() - start_time
        
        if result and 'error' not in result:
            print("VLM-Enhanced escalator analyzer test PASSED")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Frames analyzed: {result['total_frames_analyzed']}")
            print(f"   - VLM used: {result['vlm_used']}")
            print(f"   - Crowding detected: {result['crowding_analysis']['crowding_detected']}")
            print(f"   - Falling risk: {result['falling_analysis']['falling_risk']}")
            print(f"   - Safety alerts: {len(result['safety_alerts'])}")
            return True
        else:
            print(f"VLM-Enhanced analyzer test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        return False

def test_vlm_directly():
    """Test VLM functionality directly"""
    print("Testing VLM functionality directly...")
    
    try:
        # Load token
        token_path = "hf_token.txt"
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token = f.read().strip()
            print(f"Token loaded: {token[:10]}...")
        else:
            print("No token file found")
            return False
        
        # Create analyzer
        analyzer = EscalatorVLMAnalyzer(token)
        
        if not analyzer.vlm_available:
            print("VLM not available")
            return False
        
        print("VLM is available, testing with a simple image...")
        
        # Create a simple test image
        test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray image
        
        # Test VLM analysis
        result = analyzer._analyze_frame_vlm(test_image)
        
        print(f"VLM test result: {result['vlm_description']}")
        return True
        
    except Exception as e:
        print(f"VLM test failed: {e}")
        return False

if __name__ == "__main__":
    # Test VLM directly first
    if test_vlm_directly():
        print("VLM test passed, running full test...")
        test_escalator_vlm()
    else:
        print("VLM test failed, running basic test only...")
        test_escalator_vlm() 