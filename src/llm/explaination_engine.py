from typing import Dict, Any

class LLMExplanationEngine:
    def generate_advisory(self, pipeline_payload: Dict[str, Any], lang: str = "en") -> Dict[str, str]:
        crop = pipeline_payload["request"]["crop"]
        city = pipeline_payload["request"]["city"]
        gap = pipeline_payload["market_gap"]
        forecast = pipeline_payload["forecast"]
        opp = pipeline_payload["opportunity"]

        if lang == "hi":
            summary = (
                f"{city} में {crop} की मांग में भारी उछाल दर्ज किया गया है। "
                f"पूर्वानुमानित मांग {forecast['dynamic_demand_kg']:,} किग्रा है, जो अपेक्षित आपूर्ति ({forecast['forecast_supply_kg']:,} किग्रा) से "
                f"{gap['gap_percentage']}% अधिक है। अवसर स्कोर {opp['score']}/100 है। "
                f"किसानों को तुरंत अपनी उपज बाज़ार में भेजने की सलाह दी जाती है।"
            )
            action = "अपनी उपज निकटतम FPO या बाज़ार केंद्र पर पंजीकृत करें।"
        else:
            summary = (
                f"High commercial opportunity detected for {crop} in {city}. "
                f"Projected demand of {forecast['dynamic_demand_kg']:,} kg exceeds expected mandi supply ({forecast['forecast_supply_kg']:,} kg) "
                f"by {gap['gap_percentage']}%. Opportunity Score is {opp['score']}/100. "
                f"Primary drivers include live buyer order surges and favorable seasonal pricing."
            )
            action = "Dispatch ready crop volumes through matched regional FPO channels immediately."

        return {
            "summary": summary,
            "recommended_action": action,
            "language": lang
        }