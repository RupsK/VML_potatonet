"""
Thermal Imaging Domain Knowledge Injection Module
Provides targeted, actionable domain knowledge to enhance AI model prompts
"""

class ThermalKnowledgeInjector:
    def __init__(self):
        """Initialize the knowledge injector with targeted domain expertise"""
        self.domain_knowledge = {
            "Expert Thermal Analysis": {
                "description": "Professional thermal image interpretation techniques",
                "knowledge": """
                EXPERT ANALYSIS FRAMEWORK:
                - Identify temperature ranges and their significance (hot/cold zones)
                - Detect human thermal signatures (37°C core, 32-35°C skin)
                - Analyze thermal patterns for anomalies or normal heat distribution
                - Consider environmental factors affecting thermal signatures
                - Provide specific temperature measurements and their implications
                - Identify potential safety concerns or equipment issues
                """
            },
            "Human Detection Expert": {
                "description": "Advanced human detection in thermal images",
                "knowledge": """
                HUMAN DETECTION EXPERTISE:
                - Look for bright thermal signatures indicating human presence
                - Identify head/torso as brightest areas (highest temperature)
                - Count distinct thermal clusters for multiple people
                - Detect movement patterns through thermal trails
                - Analyze clothing effects on thermal visibility
                - Identify facial features through thermal patterns
                """
            },
            "Industrial Inspector": {
                "description": "Industrial thermal inspection expertise",
                "knowledge": """
                INDUSTRIAL INSPECTION EXPERTISE:
                - Identify hot spots indicating electrical or mechanical problems
                - Detect temperature variations suggesting equipment wear
                - Analyze thermal gradients for heat flow patterns
                - Identify insulation failures or air leaks (cold spots)
                - Assess safety risks from high-temperature areas
                - Provide maintenance recommendations based on thermal patterns
                """
            },
            "Security Analyst": {
                "description": "Security and surveillance thermal analysis",
                "knowledge": """
                SECURITY ANALYSIS EXPERTISE:
                - Detect human presence in low-light or obscured conditions
                - Identify unauthorized access through thermal signatures
                - Analyze crowd patterns and behavior through thermal clusters
                - Detect concealed objects through thermal anomalies
                - Assess perimeter security through thermal monitoring
                - Identify potential threats through unusual thermal patterns
                """
            },
            "Medical Thermal Expert": {
                "description": "Medical thermal imaging interpretation",
                "knowledge": """
                MEDICAL THERMAL EXPERTISE:
                - Analyze body temperature distribution for health indicators
                - Identify inflammation through localized hot spots
                - Detect circulation issues through temperature variations
                - Assess fever through elevated body temperature patterns
                - Monitor wound healing through temperature changes
                - Analyze stress responses through facial thermal patterns
                """
            },
            "Environmental Monitor": {
                "description": "Environmental thermal monitoring expertise",
                "knowledge": """
                ENVIRONMENTAL MONITORING EXPERTISE:
                - Analyze weather effects on thermal signatures
                - Identify solar heating patterns and urban heat islands
                - Monitor water temperature variations and flow patterns
                - Assess vegetation health through thermal stress indicators
                - Detect wildlife through thermal signatures
                - Analyze climate change impacts through thermal patterns
                """
            }
        }
    
    def get_knowledge_prompt(self, selected_knowledge):
        """Generate targeted, actionable prompt with domain knowledge"""
        if not selected_knowledge:
            return ""
        
        # Create focused, actionable instructions
        expert_instructions = []
        for knowledge_type in selected_knowledge:
            if knowledge_type in self.domain_knowledge:
                expert_instructions.append(self.domain_knowledge[knowledge_type]["knowledge"])
        
        if expert_instructions:
            return f"""
            EXPERT THERMAL ANALYSIS INSTRUCTIONS:
            {' '.join(expert_instructions)}
            
            ANALYSIS REQUIREMENTS:
            - Provide specific temperature measurements and ranges
            - Identify objects and their thermal characteristics
            - Detect anomalies or unusual patterns
            - Give professional interpretation of findings
            - Suggest practical implications or actions
            - Use technical terminology appropriate for thermal imaging
            
            TASK: {self._get_analysis_task(selected_knowledge)}
            """
        
        return ""
    
    def _get_analysis_task(self, selected_knowledge):
        """Generate specific analysis task based on selected knowledge"""
        tasks = []
        
        if "Expert Thermal Analysis" in selected_knowledge:
            tasks.append("Conduct comprehensive thermal analysis with temperature measurements")
        
        if "Human Detection Expert" in selected_knowledge:
            tasks.append("Focus on human detection and counting")
        
        if "Industrial Inspector" in selected_knowledge:
            tasks.append("Identify equipment issues and maintenance needs")
        
        if "Security Analyst" in selected_knowledge:
            tasks.append("Assess security implications and potential threats")
        
        if "Medical Thermal Expert" in selected_knowledge:
            tasks.append("Analyze health indicators and medical implications")
        
        if "Environmental Monitor" in selected_knowledge:
            tasks.append("Evaluate environmental factors and conditions")
        
        if not tasks:
            tasks.append("Analyze thermal patterns and provide expert interpretation")
        
        return " | ".join(tasks)
    
    def get_available_knowledge(self):
        """Get list of available knowledge domains"""
        return list(self.domain_knowledge.keys())
    
    def get_knowledge_description(self, knowledge_type):
        """Get description of a specific knowledge domain"""
        if knowledge_type in self.domain_knowledge:
            return self.domain_knowledge[knowledge_type]["description"]
        return "Unknown knowledge domain"
    
    def create_expert_prompt(self, base_prompt, selected_knowledge):
        """Create a targeted, expert-level prompt with injected knowledge"""
        knowledge_prompt = self.get_knowledge_prompt(selected_knowledge)
        
        if knowledge_prompt:
            # Combine knowledge with user prompt more effectively
            return f"{knowledge_prompt}\n\nUSER REQUEST: {base_prompt}"
        else:
            return base_prompt
    
    def get_quick_analysis_prompt(self, selected_knowledge):
        """Get a quick, focused analysis prompt for immediate use"""
        if not selected_knowledge:
            return "Analyze this thermal image professionally."
        
        focus_areas = []
        for knowledge_type in selected_knowledge:
            if knowledge_type == "Expert Thermal Analysis":
                focus_areas.append("temperature patterns and measurements")
            elif knowledge_type == "Human Detection Expert":
                focus_areas.append("human presence and detection")
            elif knowledge_type == "Industrial Inspector":
                focus_areas.append("equipment and safety issues")
            elif knowledge_type == "Security Analyst":
                focus_areas.append("security implications")
            elif knowledge_type == "Medical Thermal Expert":
                focus_areas.append("health indicators")
            elif knowledge_type == "Environmental Monitor":
                focus_areas.append("environmental factors")
        
        if focus_areas:
            return f"Expert thermal analysis focusing on: {', '.join(focus_areas)}. Provide specific measurements, identify objects, detect anomalies, and give professional interpretation."
        else:
            return "Analyze this thermal image professionally."
    
    def get_context_aware_prompt(self, selected_knowledge, temperature_data=None):
        """Get context-aware prompt based on temperature analysis"""
        if not temperature_data:
            return self.get_quick_analysis_prompt(selected_knowledge)
        
        # Extract temperature information
        mean_temp = temperature_data.get('mean_temperature', 0)
        max_temp = temperature_data.get('max_temperature', 0)
        min_temp = temperature_data.get('min_temperature', 0)
        temp_range = max_temp - min_temp
        
        # Determine context based on temperature characteristics
        context_insights = []
        
        # Human detection context
        if "Human Detection Expert" in selected_knowledge:
            if 30 <= mean_temp <= 40:
                context_insights.append("Temperature range suggests human presence")
            elif mean_temp > 40:
                context_insights.append("High temperatures may indicate equipment or fire")
            elif mean_temp < 20:
                context_insights.append("Low temperatures suggest cold environment or objects")
        
        # Industrial context
        if "Industrial Inspector" in selected_knowledge:
            if max_temp > 100:
                context_insights.append("High temperature areas require safety assessment")
            if temp_range > 50:
                context_insights.append("Large temperature variations suggest equipment issues")
        
        # Medical context
        if "Medical Thermal Expert" in selected_knowledge:
            if 35 <= mean_temp <= 40:
                context_insights.append("Temperature range suitable for medical analysis")
            elif mean_temp > 40:
                context_insights.append("Elevated temperatures may indicate fever or inflammation")
        
        # Security context
        if "Security Analyst" in selected_knowledge:
            if temp_range > 30:
                context_insights.append("Significant temperature variations suggest activity")
            if mean_temp > 35:
                context_insights.append("Warm areas may indicate human presence")
        
        # Create context-aware prompt
        base_prompt = self.get_quick_analysis_prompt(selected_knowledge)
        
        if context_insights:
            context_text = f" Context: {'; '.join(context_insights)}. Temperature range: {min_temp:.1f}-{max_temp:.1f}°C, Mean: {mean_temp:.1f}°C."
            return base_prompt + context_text
        
        return base_prompt

# Example usage
if __name__ == "__main__":
    injector = ThermalKnowledgeInjector()
    
    # Test targeted knowledge injection
    test_knowledge = ["Expert Thermal Analysis", "Human Detection Expert"]
    base_prompt = "Analyze this thermal image for human presence and temperature patterns."
    
    expert_prompt = injector.create_expert_prompt(base_prompt, test_knowledge)
    print("Enhanced Prompt:")
    print(expert_prompt)
    
    print("\nQuick Analysis Prompt:")
    quick_prompt = injector.get_quick_analysis_prompt(test_knowledge)
    print(quick_prompt) 