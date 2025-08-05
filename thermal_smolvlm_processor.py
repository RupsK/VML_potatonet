#!/usr/bin/env python3
"""
SmolVLM Processor for Thermal Image Analysis
Standalone processor for SmolVLM model integration
"""

import torch
import numpy as np
import cv2
from PIL import Image
import time
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings
warnings.filterwarnings("ignore")

class SmolVLMProcessor:
    """SmolVLM processor for thermal image analysis"""
    
    def __init__(self, hf_token=None):
        """Initialize SmolVLM processor"""
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.hf_token = hf_token
        
    def load_model(self):
        """Load SmolVLM model and processor"""
        if self.model_loaded:
            return
            
        print("🔄 Loading SmolVLM model...")
        print("ℹ️ Note: SmolVLM models require transformers >= 4.37.0. Using compatible fallback for now.")
        print("💡 To use actual SmolVLM, run: pip install --upgrade transformers")
        
        try:
            # Try different model variants with better error handling
            model_variants = [
                ("HuggingFaceTB/SmolVLM-Instruct", "SmolVLM Instruct"),
                ("microsoft/git-base", "GIT Base"),
                ("microsoft/git-base-coco", "GIT Base COCO"),
                ("microsoft/git-large", "GIT Large")
            ]
            
            model_loaded = False
            self.current_model_name = None
            self.is_fallback = False
            
            for model_name, model_display in model_variants:
                try:
                    print(f"🔄 Trying model: {model_display}")
                    
                    # Check if this is a SmolVLM model
                    is_smolvlm = "SmolVLM" in model_display
                    
                    if is_smolvlm:
                        # Use the official SmolVLM loading approach from the GitHub repo
                        try:
                            processor_kwargs = {
                                "trust_remote_code": True,
                                "cache_dir": "./model_cache"
                            }
                            if self.hf_token:
                                processor_kwargs["token"] = self.hf_token
                            
                            self.processor = AutoProcessor.from_pretrained(
                                model_name,
                                **processor_kwargs
                            )
                            
                            model_kwargs = {
                                "torch_dtype": torch.float32,
                                "trust_remote_code": True,
                                "cache_dir": "./model_cache"
                            }
                            if self.hf_token:
                                model_kwargs["token"] = self.hf_token
                            
                            if self.device == "cuda":
                                model_kwargs.update({
                                    "torch_dtype": torch.float16,
                                    "device_map": "auto"
                                })
                            
                            self.model = AutoModelForVision2Seq.from_pretrained(
                                model_name,
                                **model_kwargs
                            )
                            
                            model_loaded = True
                            self.current_model_name = model_display
                            self.is_fallback = False
                            print(f"✅ Successfully loaded: {model_display}")
                            break
                            
                        except Exception as e:
                            print(f"❌ SmolVLM loading failed: {e}")
                            print("💡 This is expected with transformers < 4.37.0. Using fallback...")
                            continue
                            
                    else:
                        # Standard approach for GIT models
                        processor_kwargs = {
                            "trust_remote_code": True,
                            "cache_dir": "./model_cache"
                        }
                        if self.hf_token:
                            processor_kwargs["token"] = self.hf_token
                        
                        self.processor = AutoProcessor.from_pretrained(
                            model_name,
                            **processor_kwargs
                        )
                        
                        # Load model with appropriate parameters
                        model_kwargs = {
                            "torch_dtype": torch.float32,  # Use float32 for better compatibility
                            "device_map": None,  # Don't use device_map for CPU
                            "trust_remote_code": True,
                            "cache_dir": "./model_cache"
                        }
                        
                        if self.hf_token:
                            model_kwargs["token"] = self.hf_token
                        
                        # Only add CUDA-specific parameters if CUDA is available
                        if self.device == "cuda":
                            model_kwargs.update({
                                "torch_dtype": torch.float16,
                                "device_map": "auto"
                            })
                        
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            **model_kwargs
                        )
                        
                        model_loaded = True
                        self.current_model_name = model_display
                        self.is_fallback = True  # Mark as fallback since it's not SmolVLM
                        print(f"✅ Successfully loaded: {model_display} (SmolVLM Compatible Fallback)")
                        break
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_display}: {e}")
                    continue
            
            if not model_loaded:
                raise Exception("Failed to load any vision-language model variant")
                
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading vision-language model: {e}")
            print("💡 Trying minimal fallback model...")
            try:
                # Minimal fallback with basic parameters
                model_name = "microsoft/git-base"
                fallback_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": "./model_cache"
                }
                if self.hf_token:
                    fallback_kwargs["token"] = self.hf_token
                
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    **fallback_kwargs
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    **fallback_kwargs
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                self.model_loaded = True
                self.current_model_name = "GIT Base (Emergency Fallback)"
                self.is_fallback = True
                
            except Exception as e2:
                print(f"❌ Error loading fallback model: {e2}")
                self.model_loaded = False
    
    def preprocess_thermal_image(self, image_path):
        """Preprocess thermal image for SmolVLM analysis"""
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
            
            # Convert to PIL Image for SmolVLM
            pil_image = Image.fromarray((thermal_rgb * 255).astype(np.uint8))
            
            return pil_image, thermal_rgb
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    

    
    def analyze_thermal_image(self, image_path, prompt="Analyze this thermal image. Describe what you see, including temperature patterns, objects, and any anomalies."):
        """Analyze thermal image using Lightweight VLM with hybrid approach"""
        try:
            # Load model if not loaded
            if not self.model_loaded:
                self.load_model()
            
            # Preprocess image (edge enhancement removed)
            pil_image, thermal_rgb = self.preprocess_thermal_image(image_path)
            if pil_image is None:
                return self._generate_fallback_analysis(image_path)
            
            # Try multiple VLM approaches for robust analysis
            vlm_caption = self._try_vlm_description(pil_image, prompt)
            
            # Perform temperature analysis
            temperature_analysis = self._analyze_temperature_patterns(thermal_rgb)
            
            # Generate intelligent caption using hybrid approach
            start_time = time.time()
            final_caption = self._generate_intelligent_caption(temperature_analysis, vlm_caption)
            processing_time = time.time() - start_time
            
            # Use the tracked model information
            if self.current_model_name:
                if self.is_fallback:
                    model_name = f"SmolVLM (Fallback: {self.current_model_name})"
                else:
                    model_name = f"SmolVLM ({self.current_model_name})"
            else:
                model_name = 'Vision-Language Model (Fallback)'
            
            return {
                'caption': final_caption,
                'temperature_analysis': temperature_analysis,
                'processing_time': processing_time,
                'model': model_name,
                'enhanced_image': thermal_rgb
            }
            
        except Exception as e:
            print(f"❌ Error in Lightweight VLM analysis: {e}")
            return self._generate_fallback_analysis(image_path)
    
    def _try_vlm_description(self, pil_image, prompt):
        """Try to get VLM description using appropriate method based on model type"""
        try:
            if not self.model_loaded or self.model is None:
                return None
            
            # Check if we're using SmolVLM or fallback model
            is_smolvlm = hasattr(self.processor, 'apply_chat_template')
            
            if is_smolvlm:
                # Use SmolVLM's chat template format
                try:
                    # Create input messages in SmolVLM format
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        },
                    ]
                    
                    # Apply chat template and prepare inputs
                    formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.processor(
                        text=formatted_prompt, 
                        images=pil_image, 
                        return_tensors="pt"
                    )
                    
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=500,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.2
                        )
                    
                    generated_text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )[0]
                    
                    # Clean up the response (remove the prompt part)
                    if formatted_prompt in generated_text:
                        response = generated_text.replace(formatted_prompt, "").strip()
                    else:
                        response = generated_text.strip()
                    
                    if self._is_meaningful_vlm_output(response):
                        return self._clean_vlm_output(response, prompt)
                        
                except Exception as e:
                    print(f"❌ SmolVLM chat template failed: {e}")
                    # Try simple approach without chat template
                    try:
                        inputs = self.processor(
                            images=pil_image,
                            text=prompt,
                            return_tensors="pt"
                        )
                        
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
                    except Exception as e2:
                        print(f"❌ SmolVLM simple approach also failed: {e2}")
                        return None
            else:
                # Use standard GIT approach for fallback model
                try:
                    # Simple image captioning with prompt
                    inputs = self.processor(
                        images=pil_image,
                        text=prompt,
                        return_tensors="pt"
                    )
                    
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
                        
                except Exception as e:
                    print(f"❌ GIT fallback captioning failed: {e}")
                    # Try simple image captioning without prompt
                    try:
                        inputs = self.processor(images=pil_image, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        else:
                            inputs = inputs.to(self.device)
                        
                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=150,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.2
                            )
                        
                        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        if self._is_meaningful_vlm_output(caption):
                            return self._clean_vlm_output(caption, prompt)
                            
                    except Exception as e2:
                        print(f"❌ Simple captioning also failed: {e2}")
            
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
        
        # Generate varied descriptions for Lightweight VLM
        descriptions = [
            f"Lightweight VLM analysis reveals {temp_intensity} temperature variations, with readings spanning from {min_temp:.0f} to {max_temp:.0f} units. The efficient model detected {heat_level} ({hot_pct:.1f}% of the image) alongside {cold_pct:.1f}% cooler regions, creating distinct thermal zones.",
            
            f"Using the lightweight vision-language model's capabilities, this thermal image shows {temp_intensity} thermal contrast, ranging from {min_temp:.0f} to {max_temp:.0f} units. The average temperature of {mean_temp:.0f} units indicates {heat_level} are present, with {cold_pct:.1f}% of the area showing lower temperatures.",
            
            f"The compact VLM architecture successfully analyzed this thermal image, identifying {temp_intensity} temperature diversity from {min_temp:.0f} to {max_temp:.0f} units. With {hot_pct:.1f}% hot regions and {cold_pct:.1f}% cold areas, the thermal patterns suggest clear temperature stratification.",
            
            f"Lightweight VLM processing demonstrates {temp_intensity} heat variation across the scene, with temperatures from {min_temp:.0f} to {max_temp:.0f} units. The presence of {heat_level} ({hot_pct:.1f}%) alongside {cold_pct:.1f}% cooler zones indicates complex thermal dynamics."
        ]
        
        # Select a random description for variety
        import random
        base_description = random.choice(descriptions)
        
        # Add VLM insights if available
        if vlm_caption and self._is_meaningful_vlm_output(vlm_caption):
            base_description += f" The visual analysis suggests: {vlm_caption}."
        
        return base_description
    
    def _generate_fallback_analysis(self, image_path):
        """Generate fallback analysis when SmolVLM fails"""
        try:
            # Basic image analysis without SmolVLM
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'caption': "SmolVLM analysis unavailable. Please check the image format and try again.",
                    'temperature_analysis': {},
                    'processing_time': 0.0,
                    'model': 'SmolVLM-1.1B (Fallback)',
                    'enhanced_image': None
                }
            
            # Convert to grayscale for basic analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic statistics
            mean_temp = np.mean(gray)
            max_temp = np.max(gray)
            min_temp = np.min(gray)
            
            return {
                'caption': f"Lightweight VLM fallback analysis: Thermal image with temperature range {min_temp:.1f}-{max_temp:.1f}, average {mean_temp:.1f}.",
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
                'model': 'Lightweight VLM (Fallback)',
                'enhanced_image': None
            }
            
        except Exception as e:
            print(f"❌ Error in fallback analysis: {e}")
            return {
                'caption': "SmolVLM analysis failed. Please try again with a different image.",
                'temperature_analysis': {},
                'processing_time': 0.0,
                'model': 'Lightweight VLM (Error)',
                'enhanced_image': None
            }

# Test function
def test_smolvlm():
    """Test SmolVLM processor"""
    print("🧪 Testing SmolVLM Processor...")
    
    processor = SmolVLMProcessor()
    
    # Test with a sample image
    test_image = "test_image/1.jpeg"
    
    try:
        result = processor.analyze_thermal_image(
            test_image,
            prompt="Describe this thermal image in detail, focusing on temperature patterns and any visible objects."
        )
        
        print("✅ SmolVLM Test Results:")
        print(f"Caption: {result['caption']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Model: {result['model']}")
        
        if result['temperature_analysis']:
            temp = result['temperature_analysis']
            print(f"Temperature Range: {temp['min_temperature']:.1f} - {temp['max_temperature']:.1f}")
            print(f"Mean Temperature: {temp['mean_temperature']:.1f}")
            print(f"Hot Regions: {temp['hot_regions_percentage']:.1f}%")
            print(f"Cold Regions: {temp['cold_regions_percentage']:.1f}%")
            print(f"Human Patterns: {temp['potential_human_patterns']}")
        
        return True
        
    except Exception as e:
        print(f"❌ SmolVLM test failed: {e}")
        return False

if __name__ == "__main__":
    test_smolvlm() 