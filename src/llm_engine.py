# src/llm_engine.py
import datetime
import os
import json
import logging
from typing import Dict, Any

class LLMMarketIntelligence:
    """Uses LLM reasoning for dynamic festival detection and multilingual farmer advisories."""

    @classmethod
    def analyze_dynamic_events_and_demand(
        cls, commodity: str, location_name: str, target_date: datetime.date = None
    ) -> Dict[str, Any]:
        """Queries LLM to reason about local festivals, fasts, and crop demand shifts for ANY location."""
        if target_date is None:
            target_date = datetime.date.today()

        date_str = target_date.strftime("%B %d, %Y")
        
        prompt = f"""
        You are an expert Indian Agricultural Economist and Cultural Calendar Analyst.
        Target Date: {date_str}
        Location: {location_name}, India
        Commodity: {commodity}

        Analyze if there are any active or upcoming (within 7 days) national, regional, or minor Indian festivals, 
        fasting periods (e.g., Navratri, Sawan, Chhath, Ekadashi, Ramzan), wedding seasons, or local events in this region.

        Return ONLY a JSON object with this exact structure:
        {{
            "event_name": "<Name of active festival/event or 'Standard Period'>",
            "demand_impact_percentage": <integer between -50 and +50 representing expected demand shift>,
            "cultural_reasoning": "<Short 1-sentence explanation of why this commodity demand changes during this event>"
        }}
        """

        # Call Gemini API if GOOGLE_API_KEY is configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                logging.warning(f"Gemini API error: {e}. Falling back to Rule Engine.")

        # Fallback Dynamic Calendar Reasoning Engine
        month, day = target_date.month, target_date.day
        if month in [9, 10]: # Autumn Festival / Fasting Season
            if commodity.lower() in ["potato", "tomato"]:
                return {
                    "event_name": "Navratri & Festival Season",
                    "demand_impact_percentage": 20,
                    "cultural_reasoning": "Fasting and feast preparations increase household consumption of potatoes and tomatoes."
                }
            elif commodity.lower() == "onion":
                return {
                    "event_name": "Navratri Fasting Period",
                    "demand_impact_percentage": -15,
                    "cultural_reasoning": "Dietary restrictions during holy fasts reduce raw onion consumption."
                }

        return {
            "event_name": "Standard Agricultural Period",
            "demand_impact_percentage": 0,
            "cultural_reasoning": "Normal baseline consumption patterns in effect."
        }

    @classmethod
    def generate_multilingual_advisory(
        cls, commodity: str, location: str, gap_kg: float, severity: str, reasoning: str, lang: str = "hi"
    ) -> str:
        """Generates localized action advisories for farmers and FPOs in English or Hindi."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                prompt = f"""
                Write a 2-sentence actionable advisory for a farmer/FPO in {location}.
                Language: {'Hindi (in Devanagari script)' if lang=='hi' else 'English'}
                Commodity: {commodity}
                Supply Gap Deficit: {gap_kg:.0f} kg ({severity} severity)
                Market Context: {reasoning}
                
                Keep the tone urgent, practical, and clear.
                """
                res = model.generate_content(prompt)
                return res.text.strip()
            except Exception as e:
                logging.warning(f"Gemini advisory error: {e}")

        # Built-in Fallback Statements
        if lang == "hi":
            return f"⚠️ {location} में {commodity} की {gap_kg:.0f} किग्रा की भारी मांग है। किसान भाइयों को सलाह दी जाती है कि वे कटाई में तेजी लाएं ताकि मंडी में उचित मूल्य प्राप्त हो सके।"
        else:
            return f"⚠️ High demand surge of {gap_kg:.0f} kg detected for {commodity} in {location}. Farmers are advised to accelerate harvest to capture premium market pricing."