import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import google.generativeai as genai
from datetime import datetime

# Pydantic models for validation
class CallSummary(BaseModel):
    summary: str = Field(..., max_length=500)
    key_points: List[str] = Field(..., max_items=5)
    outcome: str = Field(..., max_length=200)
    follow_up_recommendations: List[str] = Field(..., max_items=3)

class CallTopic(BaseModel):
    primary_topic: str = Field(..., max_length=100)
    category: str = Field(..., max_length=100)
    sub_category: str = Field(..., max_length=100)

class AgentCoaching(BaseModel):
    strengths: List[str] = Field(..., max_items=3)
    improvement_areas: List[str] = Field(..., max_items=3)
    specific_recommendations: List[str] = Field(..., max_items=4)
    skill_development_focus: List[str] = Field(..., max_items=3)

class TranscriptAnalysis(BaseModel):
    call_summary: CallSummary
    call_topic: CallTopic
    agent_coaching: AgentCoaching

class KPIExtractor:
    def __init__(self, api_key: str):
        """Initialize Gemini API and configure the model"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def create_prompt(self, transcript: str) -> str:
        """Create a structured prompt for KPI extraction"""
        return f"""
        Analyze this call transcript and provide a structured analysis in the exact JSON format specified below.
        Keep responses concise, specific, and actionable.

        Guidelines:
        - Call summary should be factual and highlight key interactions
        - Topics and categories should match standard business taxonomies
        - Coaching points should be specific and actionable
        - All responses must follow the exact structure specified
        - Ensure all lists have the specified maximum number of items
        - All text fields must be clear, professional, and free of fluff

        Transcript:
        {transcript}

        Required Output Structure:
        {{
            "call_summary": {{
                "summary": "3-4 line overview of the call",
                "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
                "outcome": "Clear statement of call resolution",
                "follow_up_recommendations": ["Rec 1", "Rec 2", "Rec 3"]
            }},
            "call_topic": {{
                "primary_topic": "Main topic of discussion",
                "category": "Business category",
                "sub_category": "Specific sub-category"
            }},
            "agent_coaching": {{
                "strengths": ["Strength 1", "Strength 2", "Strength 3"],
                "improvement_areas": ["Area 1", "Area 2", "Area 3"],
                "specific_recommendations": ["Rec 1", "Rec 2", "Rec 3", "Rec 4"],
                "skill_development_focus": ["Skill 1", "Skill 2", "Skill 3"]
            }}
        }}

        Rules:
        1. Maintain exact JSON structure
        2. No additional fields or comments
        3. No markdown formatting
        4. Ensure all arrays have the exact number of items specified
        5. Keep all text concise and professional
        """

    def validate_response(self, response_json: Dict) -> TranscriptAnalysis:
        """Validate the response using Pydantic models"""
        return TranscriptAnalysis(**response_json)

    def extract_kpis(self, transcript: str) -> Optional[Dict]:
        """
        Extract KPIs from transcript using Gemini API
        
        Args:
            transcript (str): Call transcript text
            
        Returns:
            Dict: Structured KPI data or None if extraction fails
        """
        try:
            # Generate prompt
            prompt = self.create_prompt(transcript)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_json = json.loads(response.text)
            
            # Validate response structure
            validated_response = self.validate_response(response_json)
            
            return validated_response.dict()
            
        except Exception as e:
            print(f"Error extracting KPIs: {str(e)}")
            return None

class KPIProcessor:
    def __init__(self, extractor: KPIExtractor):
        self.extractor = extractor
        
    def process_batch(self, transcripts: List[str]) -> List[Dict]:
        """Process a batch of transcripts"""
        results = []
        for transcript in transcripts:
            kpis = self.extractor.extract_kpis(transcript)
            if kpis:
                results.append(kpis)
        return results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"kpi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Initialize extractor with your API key
    api_key = "your_gemini_api_key"
    extractor = KPIExtractor(api_key)
    
    # Sample transcript
    transcript = """
    [Your transcript text here]
    """
    
    # Process single transcript
    kpis = extractor.extract_kpis(transcript)
    
    if kpis:
        print("Extracted KPIs:")
        print(json.dumps(kpis, indent=2))
    
    # Process batch of transcripts
    processor = KPIProcessor(extractor)
    transcripts = [transcript, transcript]  # Add more transcripts as needed
    results = processor.process_batch(transcripts)
    
    # Save results
    processor.save_results(results)
