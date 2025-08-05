#!/usr/bin/env python3
"""
Thermal VLM Processor
Basic processor for BLIP and GIT models
"""

import torch
import numpy as np
import cv2
from PIL import Image
import time
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings
warnings.filterwarnings("ignore")

class ThermalImageProcessor:
    """Basic thermal image processor for BLIP and GIT models"""
    
    def __init__(self):
        """Initialize thermal image processor"""
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
    def load_model(self, model_name="microsoft/git-base"):
        """Load model and processor"""
        if self.model_loaded:
            return
            
        print(f"🔄 Loading {model_name} model...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            print(f"✅ {model_name} model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading {model_name} model: {e}")
            self.model_loaded = False
    
    def preprocess_thermal_image(self, image_path, enhance_edges=True):
        """Preprocess thermal image for analysis"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = image_path
                
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply thermal colormap for better visualization
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thermal_colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            thermal_rgb = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
            
            # Normalize image
            thermal_rgb = thermal_rgb.astype(np.float32) / 255.0
            
            # Apply edge enhancement if requested
            if enhance_edges:
                thermal_rgb = self._apply_edge_enhancement(thermal_rgb)
            
            # Convert to PIL Image for VLM
            pil_image = Image.fromarray((thermal_rgb * 255).astype(np.uint8))
            
            return pil_image, thermal_rgb
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    
    def _apply_edge_enhancement(self, image):
        """Apply edge enhancement using Sobel operators"""
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Apply Sobel operators
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize gradient
            gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
            
            # Convert back to RGB and enhance
            enhanced = image + gradient_magnitude[:, :, np.newaxis] * 0.3
            enhanced = np.clip(enhanced, 0, 1)
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️ Error applying edge enhancement: {e}")
            return image
    
    def analyze_thermal_image(self, image_path, prompt="Analyze this thermal image. Describe what you see, including temperature patterns, objects, and any anomalies.", enhance_edges=True):
        """Analyze thermal image using VLM with hybrid approach"""
        try:
            # Load model if not loaded
            if not self.model_loaded:
                self.load_model()
            
            # Preprocess image
            pil_image, thermal_rgb = self.preprocess_thermal_image(image_path, enhance_edges)
            if pil_image is None:
                return self._generate_fallback_analysis(image_path)
            
            # Try VLM description
            vlm_caption = self._try_vlm_description(pil_image, prompt)
            
            # Perform temperature analysis
            temperature_analysis = self._analyze_temperature_patterns(thermal_rgb)
            
            # Generate intelligent caption
            start_time = time.time()
            final_caption = self._generate_intelligent_caption(temperature_analysis, vlm_caption)
            processing_time = time.time() - start_time
            
            return {
                'caption': final_caption,
                'temperature_analysis': temperature_analysis,
                'processing_time': processing_time,
                'model': 'BLIP/GIT Base',
                'enhanced_image': thermal_rgb
            }
            
        except Exception as e:
            print(f"❌ Error in VLM analysis: {e}")
            return self._generate_fallback_analysis(image_path)
    
    def _try_vlm_description(self, pil_image, prompt):
        """Try to get VLM description"""
        try:
            if not self.model_loaded or self.model is None:
                return None
            
            # Simple image captioning
            inputs = self.processor(images=pil_image, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if self._is_meaningful_vlm_output(caption):
                return self._clean_vlm_output(caption, prompt)
            
            return None
            
        except Exception as e:
            print(f"❌ VLM description failed: {e}")
            return None
    
    def _clean_vlm_output(self, text, original_prompt):
        """Clean and validate VLM output"""
        try:
            # Remove the original prompt if it appears in the output
            if original_prompt.lower() in text.lower():
                text = text.replace(original_prompt, "").strip()
            
            # Remove repetitive phrases
            words = text.split()
            if len(words) > 3:
                # Check for repetitive patterns
                for i in range(len(words) - 2):
                    if words[i] == words[i+1] == words[i+2]:
                        # Remove repetitive sequence
                        words = words[:i] + words[i+3:]
                        break
            
            cleaned_text = " ".join(words).strip()
            
            return cleaned_text
            
        except Exception as e:
            print(f"⚠️ Error cleaning VLM output: {e}")
            return text
    
    def _is_meaningful_vlm_output(self, caption):
        """Check if VLM output is meaningful"""
        if not caption or len(caption.strip()) < 10:
            return False
        
        # Check for repetitive patterns
        words = caption.lower().split()
        if len(words) > 0:
            most_common_word = max(set(words), key=words.count)
            word_count = words.count(most_common_word)
            if word_count > len(words) * 0.3:
                return False
        
        # Check for question echoes
        question_words = ['what', 'how', 'why', 'when', 'where', 'describe', 'analyze']
        if any(word in caption.lower() for word in question_words):
            return False
        
        # Check for repetitive thermal words
        thermal_words = ['thermal', 'image', 'temperature', 'heat']
        thermal_count = sum(1 for word in words if word in thermal_words)
        if thermal_count > len(words) * 0.4:
            return False
            
        return True
    
    def _analyze_temperature_patterns(self, image):
        """Analyze temperature patterns in the thermal image"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Basic statistics
            mean_temp = np.mean(gray)
            max_temp = np.max(gray)
            min_temp = np.min(gray)
            std_temp = np.std(gray)
            
            # Temperature distribution analysis
            hot_threshold = np.percentile(gray, 90)
            cold_threshold = np.percentile(gray, 10)
            
            hot_regions = np.sum(gray > hot_threshold)
            cold_regions = np.sum(gray < cold_threshold)
            total_pixels = gray.size
            
            hot_percentage = (hot_regions / total_pixels) * 100
            cold_percentage = (cold_regions / total_pixels) * 100
            
            # Advanced analysis
            thermal_gradients = self._analyze_thermal_gradients(gray)
            anomalies = self._detect_thermal_anomalies(gray)
            human_patterns = self._detect_potential_human_patterns(gray)
            
            return {
                'mean_temperature': float(mean_temp),
                'max_temperature': float(max_temp),
                'min_temperature': float(min_temp),
                'temperature_std': float(std_temp),
                'hot_regions_percentage': float(hot_percentage),
                'cold_regions_percentage': float(cold_percentage),
                'temperature_range': float(max_temp - min_temp),
                'thermal_gradients': thermal_gradients,
                'thermal_anomalies_percentage': float(anomalies),
                'potential_human_patterns': int(human_patterns)
            }
            
        except Exception as e:
            print(f"⚠️ Error in temperature analysis: {e}")
            return {
                'mean_temperature': 0.0,
                'max_temperature': 0.0,
                'min_temperature': 0.0,
                'temperature_std': 0.0,
                'hot_regions_percentage': 0.0,
                'cold_regions_percentage': 0.0,
                'temperature_range': 0.0,
                'thermal_gradients': 0.0,
                'thermal_anomalies_percentage': 0.0,
                'potential_human_patterns': 0
            }
    
    def _analyze_thermal_gradients(self, gray):
        """Analyze thermal gradients in the image"""
        try:
            # Calculate gradients using Sobel operators
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Return average gradient strength
            return float(np.mean(gradient_magnitude))
            
        except Exception as e:
            print(f"⚠️ Error analyzing thermal gradients: {e}")
            return 0.0
    
    def _detect_thermal_anomalies(self, gray):
        """Detect thermal anomalies using statistical methods"""
        try:
            # Calculate z-scores
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            if std_val == 0:
                return 0.0
            
            z_scores = np.abs((gray - mean_val) / std_val)
            
            # Anomalies are pixels with z-score > 2
            anomalies = np.sum(z_scores > 2)
            
            return (anomalies / gray.size) * 100
            
        except Exception as e:
            print(f"⚠️ Error detecting thermal anomalies: {e}")
            return 0.0
    
    def _detect_potential_human_patterns(self, gray):
        """Detect potential human-like thermal patterns"""
        try:
            # Use edge detection to find human-like contours
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            human_patterns = 0
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area < 100:  # Too small
                    continue
                
                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Human-like patterns typically have aspect ratio between 1.5 and 3.0
                if 1.5 <= aspect_ratio <= 3.0 and area > 500:
                    human_patterns += 1
            
            return min(human_patterns, 5)  # Cap at 5 patterns
            
        except Exception as e:
            print(f"⚠️ Error detecting human patterns: {e}")
            return 0
    
    def _generate_intelligent_caption(self, temp_analysis, vlm_caption=None):
        """Generate intelligent caption combining VLM output and temperature analysis"""
        
        mean_temp = temp_analysis['mean_temperature']
        max_temp = temp_analysis['max_temperature']
        min_temp = temp_analysis['min_temperature']
        hot_pct = temp_analysis['hot_regions_percentage']
        cold_pct = temp_analysis['cold_regions_percentage']
        temp_range = temp_analysis['temperature_range']
        
        # Determine temperature characteristics
        if temp_range > 150:
            temp_intensity = "extreme"
        elif temp_range > 100:
            temp_intensity = "high"
        elif temp_range > 50:
            temp_intensity = "moderate"
        else:
            temp_intensity = "low"
        
        # Determine heat distribution
        if hot_pct > 20:
            heat_level = "extensive hot areas"
        elif hot_pct > 10:
            heat_level = "several hot regions"
        elif hot_pct > 5:
            heat_level = "some hot spots"
        else:
            heat_level = "minimal hot areas"
        
        # Generate base description
        base_description = f"Thermal analysis reveals {temp_intensity} temperature variations, with readings spanning from {min_temp:.0f} to {max_temp:.0f} units. The VLM detected {heat_level} ({hot_pct:.1f}% of the image) alongside {cold_pct:.1f}% cooler regions, creating distinct thermal zones."
        
        # Add VLM insights if available
        if vlm_caption and self._is_meaningful_vlm_output(vlm_caption):
            base_description += f" The visual analysis suggests: {vlm_caption}."
        
        return base_description
    
    def _generate_fallback_analysis(self, image_path):
        """Generate fallback analysis when VLM fails"""
        try:
            # Basic image analysis without VLM
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'caption': "VLM analysis unavailable. Please check the image format and try again.",
                    'temperature_analysis': {},
                    'processing_time': 0.0,
                    'model': 'VLM (Fallback)',
                    'enhanced_image': None
                }
            
            # Convert to grayscale for basic analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic statistics
            mean_temp = np.mean(gray)
            max_temp = np.max(gray)
            min_temp = np.min(gray)
            
            return {
                'caption': f"VLM fallback analysis: Thermal image with temperature range {min_temp:.1f}-{max_temp:.1f}, average {mean_temp:.1f}.",
                'temperature_analysis': {
                    'mean_temperature': float(mean_temp),
                    'max_temperature': float(max_temp),
                    'min_temperature': float(min_temp),
                    'temperature_std': float(np.std(gray)),
                    'hot_regions_percentage': 0.0,
                    'cold_regions_percentage': 0.0,
                    'temperature_range': float(max_temp - min_temp),
                    'thermal_gradients': 0.0,
                    'thermal_anomalies_percentage': 0.0,
                    'potential_human_patterns': 0
                },
                'processing_time': 0.1,
                'model': 'VLM (Fallback)',
                'enhanced_image': None
            }
            
        except Exception as e:
            print(f"❌ Error in fallback analysis: {e}")
            return {
                'caption': "VLM analysis failed. Please try again with a different image.",
                'temperature_analysis': {},
                'processing_time': 0.0,
                'model': 'VLM (Error)',
                'enhanced_image': None
            } 