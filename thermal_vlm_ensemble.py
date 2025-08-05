import torch
import cv2
import numpy as np
from PIL import Image
import time
import random
from thermal_vlm_processor import ThermalImageProcessor
from thermal_vlm_comparison import ThermalVLMComparison

class ThermalVLMEnsemble:
    def __init__(self):
        """
        Initialize ensemble system with multiple VLM models
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize individual model processors
        self.blip_base_processor = ThermalImageProcessor()
        self.blip_large_processor = ThermalImageProcessor()
        self.git_processor = ThermalImageProcessor()
        self.comparison_processor = ThermalVLMComparison()
        
        # Ensemble weights (can be adjusted based on model performance)
        self.model_weights = {
            'BLIP Base': 0.25,
            'BLIP Large': 0.30,
            'GIT Base': 0.20,
            'LLaVA-Next': 0.25
        }
        
        print("✅ Thermal VLM Ensemble initialized successfully")
    
    def preprocess_thermal_image(self, image_path, enhance_edges=True):
        """Preprocess thermal image with edge enhancement"""
        return self.comparison_processor.preprocess_thermal_image(image_path, enhance_edges)
    
    def analyze_with_ensemble(self, image_path, custom_prompt=None, ensemble_method="weighted_average"):
        """
        Analyze thermal image using ensemble of multiple VLM models
        
        Args:
            image_path: Path to the thermal image
            custom_prompt: Optional custom prompt
            ensemble_method: "weighted_average", "majority_vote", "best_model"
            
        Returns:
            dict: Ensemble analysis results
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            pil_image = self.preprocess_thermal_image(image_path)
            if pil_image is None:
                return None
            
            # Get individual model results
            results = {}
            
            # BLIP Base
            try:
                self.blip_base_processor.load_model("Salesforce/blip-image-captioning-base")
                blip_base_result = self.blip_base_processor.analyze_thermal_image(image_path, custom_prompt)
                confidence, confidence_breakdown = self._calculate_confidence_with_breakdown(blip_base_result['caption'])
                results['BLIP Base'] = {
                    'caption': blip_base_result['caption'],
                    'confidence': confidence,
                    'confidence_breakdown': confidence_breakdown,
                    'processing_time': 0.5  # Estimated
                }
            except Exception as e:
                results['BLIP Base'] = {'caption': f"BLIP Base failed: {str(e)}", 'confidence': 0.0, 'confidence_breakdown': {}, 'processing_time': 0}
            
            # BLIP Large
            try:
                self.blip_large_processor.load_model("Salesforce/blip-image-captioning-large")
                blip_large_result = self.blip_large_processor.analyze_thermal_image(image_path, custom_prompt)
                confidence, confidence_breakdown = self._calculate_confidence_with_breakdown(blip_large_result['caption'])
                results['BLIP Large'] = {
                    'caption': blip_large_result['caption'],
                    'confidence': confidence,
                    'confidence_breakdown': confidence_breakdown,
                    'processing_time': 0.8  # Estimated
                }
            except Exception as e:
                results['BLIP Large'] = {'caption': f"BLIP Large failed: {str(e)}", 'confidence': 0.0, 'confidence_breakdown': {}, 'processing_time': 0}
            
            # GIT Base
            try:
                self.git_processor.load_model("microsoft/git-base")
                git_result = self.git_processor.analyze_thermal_image(image_path, custom_prompt)
                confidence, confidence_breakdown = self._calculate_confidence_with_breakdown(git_result['caption'])
                results['GIT Base'] = {
                    'caption': git_result['caption'],
                    'confidence': confidence,
                    'confidence_breakdown': confidence_breakdown,
                    'processing_time': 0.6  # Estimated
                }
            except Exception as e:
                results['GIT Base'] = {'caption': f"GIT Base failed: {str(e)}", 'confidence': 0.0, 'confidence_breakdown': {}, 'processing_time': 0}
            
            # LLaVA-Next (from comparison processor)
            try:
                llava_result = self.comparison_processor.analyze_with_llava_next(pil_image, custom_prompt)
                confidence, confidence_breakdown = self._calculate_confidence_with_breakdown(llava_result['caption'])
                results['LLaVA-Next'] = {
                    'caption': llava_result['caption'],
                    'confidence': confidence,
                    'confidence_breakdown': confidence_breakdown,
                    'processing_time': llava_result['processing_time']
                }
            except Exception as e:
                results['LLaVA-Next'] = {'caption': f"LLaVA-Next failed: {str(e)}", 'confidence': 0.0, 'confidence_breakdown': {}, 'processing_time': 0}
            
            # Generate ensemble result
            ensemble_caption = self._generate_ensemble_caption(results, ensemble_method)
            
            # Get temperature analysis
            img_array = np.array(pil_image)
            temp_analysis = self.comparison_processor._analyze_temperature_patterns(img_array)
            
            total_time = time.time() - start_time
            
            return {
                'ensemble_caption': ensemble_caption,
                'individual_results': results,
                'ensemble_method': ensemble_method,
                'temperature_analysis': temp_analysis,
                'image': pil_image,
                'total_processing_time': total_time,
                'model_weights': self.model_weights
            }
            
        except Exception as e:
            print(f"Ensemble analysis failed: {e}")
            return None
    
    def _calculate_confidence(self, caption):
        """Calculate confidence score for a caption using multiple factors"""
        if not caption or len(caption.strip()) < 20:
            return 0.0
        
        # Factor 1: Content Quality (30% weight)
        content_score = self._calculate_content_quality(caption)
        
        # Factor 2: Thermal Analysis Depth (25% weight)
        thermal_score = self._calculate_thermal_analysis_depth(caption)
        
        # Factor 3: Object Detection Quality (20% weight)
        object_score = self._calculate_object_detection_quality(caption)
        
        # Factor 4: Technical Accuracy (15% weight)
        technical_score = self._calculate_technical_accuracy(caption)
        
        # Factor 5: Readability (10% weight)
        readability_score = self._calculate_readability(caption)
        
        # Weighted average
        confidence = (content_score * 0.30 + 
                     thermal_score * 0.25 + 
                     object_score * 0.20 + 
                     technical_score * 0.15 + 
                     readability_score * 0.10)
        
        return min(confidence, 1.0)
    
    def _calculate_content_quality(self, caption):
        """Calculate content quality score"""
        words = caption.lower().split()
        if len(words) < 10:
            return 0.3
        
        # Penalize repetitive content
        most_common_word = max(set(words), key=words.count)
        repetition_ratio = words.count(most_common_word) / len(words)
        if repetition_ratio > 0.3:
            return 0.4
        
        # Bonus for diverse vocabulary
        unique_words = len(set(words))
        vocabulary_score = min(unique_words / len(words), 1.0)
        
        # Bonus for appropriate length (not too short, not too long)
        length_score = min(len(caption) / 300.0, 1.0)  # Optimal around 300 chars
        
        return (vocabulary_score * 0.6 + length_score * 0.4)
    
    def _calculate_thermal_analysis_depth(self, caption):
        """Calculate thermal analysis depth score"""
        caption_lower = caption.lower()
        
        # Thermal terminology
        thermal_terms = ['temperature', 'thermal', 'heat', 'hot', 'cold', 'warm', 'cool']
        thermal_count = sum(1 for term in thermal_terms if term in caption_lower)
        
        # Temperature ranges and values
        temp_patterns = ['range', 'from', 'to', 'units', 'degrees', 'celsius', 'fahrenheit']
        temp_analysis = sum(1 for pattern in temp_patterns if pattern in caption_lower)
        
        # Thermal patterns
        pattern_terms = ['pattern', 'region', 'area', 'distribution', 'gradient', 'anomaly']
        pattern_count = sum(1 for term in pattern_terms if term in caption_lower)
        
        # Calculate score
        base_score = min(thermal_count * 0.2, 0.6)
        analysis_score = min(temp_analysis * 0.15, 0.3)
        pattern_score = min(pattern_count * 0.1, 0.1)
        
        return min(base_score + analysis_score + pattern_score, 1.0)
    
    def _calculate_object_detection_quality(self, caption):
        """Calculate object detection quality score"""
        caption_lower = caption.lower()
        
        # Object detection terms
        object_terms = ['people', 'person', 'human', 'individuals', 'objects', 'building', 'structure']
        object_count = sum(1 for term in object_terms if term in caption_lower)
        
        # Specific object descriptions
        specific_objects = ['camera', 'equipment', 'device', 'vehicle', 'animal', 'furniture']
        specific_count = sum(1 for term in specific_objects if term in caption_lower)
        
        # Spatial relationships
        spatial_terms = ['in front of', 'behind', 'next to', 'near', 'far', 'left', 'right']
        spatial_count = sum(1 for term in spatial_terms if term in caption_lower)
        
        # Calculate score
        base_score = min(object_count * 0.3, 0.6)
        specific_score = min(specific_count * 0.2, 0.3)
        spatial_score = min(spatial_count * 0.1, 0.1)
        
        return min(base_score + specific_score + spatial_score, 1.0)
    
    def _calculate_technical_accuracy(self, caption):
        """Calculate technical accuracy score"""
        caption_lower = caption.lower()
        
        # Technical terms
        technical_terms = ['analysis', 'detection', 'signature', 'contrast', 'variation', 'distribution']
        technical_count = sum(1 for term in technical_terms if term in caption_lower)
        
        # Percentage mentions
        percentage_patterns = ['%', 'percent', 'percentage']
        percentage_count = sum(1 for pattern in percentage_patterns if pattern in caption_lower)
        
        # Numerical values
        import re
        numbers = re.findall(r'\d+', caption)
        number_score = min(len(numbers) * 0.1, 0.3)
        
        # Calculate score
        base_score = min(technical_count * 0.2, 0.4)
        percentage_score = min(percentage_count * 0.2, 0.3)
        
        return min(base_score + percentage_score + number_score, 1.0)
    
    def _calculate_readability(self, caption):
        """Calculate readability score"""
        # Sentence structure
        sentences = caption.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Average sentence length (optimal around 15-20 words)
        avg_sentence_length = len(caption.split()) / len(sentences)
        if 10 <= avg_sentence_length <= 25:
            length_score = 1.0
        elif 5 <= avg_sentence_length <= 30:
            length_score = 0.7
        else:
            length_score = 0.4
        
        # Punctuation and formatting
        punctuation_count = caption.count(',') + caption.count(';') + caption.count(':')
        punctuation_score = min(punctuation_count * 0.1, 0.3)
        
        return min(length_score * 0.7 + punctuation_score, 1.0)
    
    def _calculate_confidence_with_breakdown(self, caption):
        """Calculate confidence score with detailed breakdown"""
        if not caption or len(caption.strip()) < 20:
            return 0.0, {}
        
        # Calculate individual scores
        content_score = self._calculate_content_quality(caption)
        thermal_score = self._calculate_thermal_analysis_depth(caption)
        object_score = self._calculate_object_detection_quality(caption)
        technical_score = self._calculate_technical_accuracy(caption)
        readability_score = self._calculate_readability(caption)
        
        # Calculate weighted confidence
        confidence = (content_score * 0.30 + 
                     thermal_score * 0.25 + 
                     object_score * 0.20 + 
                     technical_score * 0.15 + 
                     readability_score * 0.10)
        
        # Create breakdown
        breakdown = {
            'Content Quality (30%)': f"{content_score:.3f}",
            'Thermal Analysis (25%)': f"{thermal_score:.3f}",
            'Object Detection (20%)': f"{object_score:.3f}",
            'Technical Accuracy (15%)': f"{technical_score:.3f}",
            'Readability (10%)': f"{readability_score:.3f}",
            'Overall Confidence': f"{confidence:.3f}"
        }
        
        return min(confidence, 1.0), breakdown
    
    def _generate_ensemble_caption(self, results, method="weighted_average"):
        """Generate ensemble caption using specified method"""
        
        if method == "weighted_average":
            return self._weighted_average_ensemble(results)
        elif method == "majority_vote":
            return self._majority_vote_ensemble(results)
        elif method == "best_model":
            return self._best_model_ensemble(results)
        else:
            return self._weighted_average_ensemble(results)
    
    def _weighted_average_ensemble(self, results):
        """Generate ensemble caption using weighted average approach"""
        # Get the best performing model based on confidence
        best_model = max(results.items(), key=lambda x: x[1]['confidence'])
        best_caption = best_model[1]['caption']
        best_model_name = best_model[0]
        
        # Extract key information from all models
        all_insights = self._extract_key_insights(results)
        
        # Create a natural, flowing ensemble summary
        ensemble_summary = self._create_natural_ensemble_summary(all_insights, best_model_name)
        
        return ensemble_summary
    
    def _extract_key_insights(self, results):
        """Extract key insights from all model results"""
        insights = {
            'temperature_data': [],
            'object_detection': [],
            'thermal_patterns': [],
            'technical_analysis': [],
            'spatial_info': []
        }
        
        for model_name, result in results.items():
            caption = result['caption'].lower()
            
            # Extract temperature information
            if any(word in caption for word in ['temperature', 'thermal', 'heat', 'hot', 'cold']):
                temp_info = self._extract_temperature_info(result['caption'])
                if temp_info:
                    insights['temperature_data'].append(temp_info)
            
            # Extract object detection
            if any(word in caption for word in ['people', 'person', 'human', 'individuals', 'camera', 'equipment']):
                object_info = self._extract_object_info(result['caption'])
                if object_info:
                    insights['object_detection'].append(object_info)
            
            # Extract thermal patterns
            if any(word in caption for word in ['pattern', 'region', 'area', 'distribution', 'anomaly']):
                pattern_info = self._extract_pattern_info(result['caption'])
                if pattern_info:
                    insights['thermal_patterns'].append(pattern_info)
            
            # Extract technical analysis
            if any(word in caption for word in ['analysis', 'detection', 'signature', 'contrast']):
                tech_info = self._extract_technical_info(result['caption'])
                if tech_info:
                    insights['technical_analysis'].append(tech_info)
            
            # Extract spatial information
            if any(word in caption for word in ['in front of', 'behind', 'next to', 'line', 'standing']):
                spatial_info = self._extract_spatial_info(result['caption'])
                if spatial_info:
                    insights['spatial_info'].append(spatial_info)
        
        return insights
    
    def _extract_temperature_info(self, caption):
        """Extract temperature-related information"""
        import re
        
        # Find temperature ranges
        temp_ranges = re.findall(r'(\d+)\s*to\s*(\d+)\s*units?', caption)
        if temp_ranges:
            return f"Temperature range: {temp_ranges[0][0]}-{temp_ranges[0][1]} units"
        
        # Find individual temperature values
        temps = re.findall(r'(\d+)\s*units?', caption)
        if len(temps) >= 2:
            return f"Temperature variation: {temps[0]}-{temps[-1]} units"
        
        return None
    
    def _extract_object_info(self, caption):
        """Extract object detection information"""
        objects = []
        
        if 'people' in caption or 'person' in caption or 'human' in caption:
            objects.append('human presence')
        if 'camera' in caption:
            objects.append('camera equipment')
        if 'individuals' in caption:
            objects.append('multiple individuals')
        
        if objects:
            return ', '.join(objects)
        return None
    
    def _extract_pattern_info(self, caption):
        """Extract thermal pattern information"""
        patterns = []
        
        if 'pattern' in caption:
            patterns.append('thermal patterns')
        if 'region' in caption or 'area' in caption:
            patterns.append('distinct regions')
        if 'anomaly' in caption:
            patterns.append('thermal anomalies')
        if 'distribution' in caption:
            patterns.append('temperature distribution')
        
        if patterns:
            return ', '.join(patterns)
        return None
    
    def _extract_technical_info(self, caption):
        """Extract technical analysis information"""
        tech_terms = []
        
        if 'analysis' in caption:
            tech_terms.append('comprehensive analysis')
        if 'detection' in caption:
            tech_terms.append('advanced detection')
        if 'signature' in caption:
            tech_terms.append('thermal signatures')
        if 'contrast' in caption:
            tech_terms.append('thermal contrast')
        
        if tech_terms:
            return ', '.join(tech_terms)
        return None
    
    def _extract_spatial_info(self, caption):
        """Extract spatial relationship information"""
        spatial = []
        
        if 'line' in caption:
            spatial.append('arranged in a line')
        if 'standing' in caption:
            spatial.append('standing position')
        if 'shoulder' in caption:
            spatial.append('equipment on shoulder')
        
        if spatial:
            return ', '.join(spatial)
        return None
    
    def _create_natural_ensemble_summary(self, insights, best_model_name):
        """Create a natural, flowing ensemble summary"""
        summary_parts = []
        
        # Start with a natural introduction
        summary_parts.append("Comprehensive thermal analysis reveals")
        
        # Add temperature information
        if insights['temperature_data']:
            temp_info = insights['temperature_data'][0]  # Use the first/best one
            summary_parts.append(f"{temp_info.lower()}.")
        
        # Add object detection
        if insights['object_detection']:
            object_info = insights['object_detection'][0]
            summary_parts.append(f"The scene shows {object_info}.")
        
        # Add thermal patterns
        if insights['thermal_patterns']:
            pattern_info = insights['thermal_patterns'][0]
            summary_parts.append(f"Analysis identifies {pattern_info}.")
        
        # Add technical analysis
        if insights['technical_analysis']:
            tech_info = insights['technical_analysis'][0]
            summary_parts.append(f"Advanced {tech_info} techniques were employed.")
        
        # Add spatial information
        if insights['spatial_info']:
            spatial_info = insights['spatial_info'][0]
            summary_parts.append(f"Subjects appear to be {spatial_info}.")
        
        # Combine all parts naturally
        natural_summary = " ".join(summary_parts)
        
        # Add ensemble method note
        natural_summary += f"\n\n[Ensemble analysis combining multiple AI models, with {best_model_name} providing the primary analysis]"
        
        return natural_summary
    
    def _is_different_insight(self, caption1, caption2):
        """Check if two captions provide significantly different insights"""
        # Convert to lowercase for comparison
        c1_lower = caption1.lower()
        c2_lower = caption2.lower()
        
        # Extract key words from both captions
        words1 = set(c1_lower.split())
        words2 = set(c2_lower.split())
        
        # Calculate similarity (Jaccard similarity)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return True
        
        similarity = intersection / union
        
        # If similarity is less than 0.7, consider them different
        return similarity < 0.7
    
    def _majority_vote_ensemble(self, results):
        """Generate ensemble caption using majority vote approach"""
        # Count common themes across models
        themes = {
            'human_detection': 0,
            'temperature_analysis': 0,
            'thermal_patterns': 0,
            'object_detection': 0,
            'camera_equipment': 0,
            'indoor_scene': 0
        }
        
        # Analyze all captions for common themes
        for result in results.values():
            caption = result['caption'].lower()
            
            if any(word in caption for word in ['people', 'person', 'human', 'individuals']):
                themes['human_detection'] += 1
            if any(word in caption for word in ['temperature', 'thermal', 'heat', 'hot', 'cold']):
                themes['temperature_analysis'] += 1
            if any(word in caption for word in ['pattern', 'region', 'area', 'distribution']):
                themes['thermal_patterns'] += 1
            if any(word in caption for word in ['object', 'building', 'structure', 'equipment']):
                themes['object_detection'] += 1
            if any(word in caption for word in ['camera', 'equipment', 'device']):
                themes['camera_equipment'] += 1
            if any(word in caption for word in ['room', 'indoor', 'inside', 'interior']):
                themes['indoor_scene'] += 1
        
        # Find dominant themes (appearing in at least 2 models)
        dominant_themes = [theme for theme, count in themes.items() if count >= 2]
        
        # Get the best individual caption for context
        best_model = max(results.items(), key=lambda x: x[1]['confidence'])
        best_caption = best_model[1]['caption']
        
        # Create natural majority vote description
        ensemble_description = "Consensus analysis across multiple AI models indicates "
        
        theme_descriptions = []
        if 'human_detection' in dominant_themes:
            theme_descriptions.append("clear human presence")
        if 'temperature_analysis' in dominant_themes:
            theme_descriptions.append("significant temperature variations")
        if 'thermal_patterns' in dominant_themes:
            theme_descriptions.append("complex thermal patterns")
        if 'camera_equipment' in dominant_themes:
            theme_descriptions.append("camera equipment")
        if 'indoor_scene' in dominant_themes:
            theme_descriptions.append("indoor environment")
        
        if theme_descriptions:
            ensemble_description += ", ".join(theme_descriptions) + "."
        else:
            ensemble_description += "various thermal characteristics."
        
        # Add the best individual analysis in a natural way
        ensemble_description += f" Detailed analysis confirms: {best_caption}"
        
        return ensemble_description
    
    def _best_model_ensemble(self, results):
        """Generate ensemble caption using best performing model"""
        best_model = max(results.items(), key=lambda x: x[1]['confidence'])
        best_model_name = best_model[0]
        best_caption = best_model[1]['caption']
        confidence = best_model[1]['confidence']
        
        # Create a natural description
        natural_summary = f"Primary analysis using {best_model_name} (confidence: {confidence:.2f}) reveals: {best_caption}"
        
        return natural_summary
    
    def get_ensemble_methods(self):
        """Get available ensemble methods"""
        return ["weighted_average", "majority_vote", "best_model"]
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            'BLIP Base': {
                'description': 'Fast, efficient thermal image analysis',
                'strength': 'Quick object detection',
                'weight': self.model_weights['BLIP Base']
            },
            'BLIP Large': {
                'description': 'Advanced thermal image understanding',
                'strength': 'Detailed object recognition',
                'weight': self.model_weights['BLIP Large']
            },
            'GIT Base': {
                'description': 'Alternative VLM approach',
                'strength': 'Different perspective on thermal analysis',
                'weight': self.model_weights['GIT Base']
            },
            'LLaVA-Next': {
                'description': 'Enhanced thermal analysis with advanced features',
                'strength': 'Sophisticated pattern recognition',
                'weight': self.model_weights['LLaVA-Next']
            }
        }

def main():
    """Test the ensemble system"""
    ensemble = ThermalVLMEnsemble()
    
    # Test with a sample image
    test_image = "test_image/download.jpg"  # Adjust path as needed
    
    print("Testing ensemble analysis...")
    result = ensemble.analyze_with_ensemble(test_image, ensemble_method="weighted_average")
    
    if result:
        print(f"\nEnsemble Caption: {result['ensemble_caption']}")
        print(f"Ensemble Method: {result['ensemble_method']}")
        print(f"Total Processing Time: {result['total_processing_time']:.2f}s")
        
        print("\nIndividual Model Results:")
        for model, model_result in result['individual_results'].items():
            print(f"{model}: Confidence {model_result['confidence']:.2f}")

if __name__ == "__main__":
    main() 