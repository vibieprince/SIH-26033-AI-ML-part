# src/llm_synthesizer.py
from __future__ import annotations
"""
LangChain + Gemini demand advisory synthesizer.

Key fixes vs. the original:
1. All real computed numbers (baseline_ml_kg, live_orders_kg, supply_kg,
   gap_pct, weather temp) are injected into the prompt – the model cannot
   invent them.
2. Normal / baseline conditions produce *explicit* named drivers instead of
   returning an empty list or falling through to a static JSON string.
3. The Pydantic schema uses a dedicated SignalDriver model so impact
   direction and explanation are always structured.
4. The fallback (no API key / network error) builds a fully dynamic report
   from the real numbers – no hardcoded 22/100 or 5.6% artifacts.
"""

import logging
import os
from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Pydantic schema
# ------------------------------------------------------------------ #

class SignalDriver(BaseModel):
    """A single identified demand signal."""

    driver_name: str = Field(
        description=(
            "Short, specific name for the signal, e.g. "
            "'Standard Seasonal Baseline', 'Diwali Festival Surge', "
            "'Heavy Rain / Transit Disruption', 'Active Platform Order Inflow'."
        )
    )
    impact_direction: str = Field(
        description="One of: INCREASE, DECREASE, NEUTRAL"
    )
    impact_magnitude_pct: float = Field(
        description=(
            "Estimated percentage change this driver contributes to demand. "
            "Use 0.0 for truly neutral drivers."
        )
    )
    explanation: str = Field(
        description="Concise explanation of why this signal affects demand."
    )


class DemandAdvisorySchema(BaseModel):
    """Structured AI demand intelligence report returned by the LLM."""

    predicted_demand_kg: float = Field(
        description="Final demand forecast in kilograms after incorporating all signals."
    )
    predicted_supply_kg: float = Field(
        description="Predicted available mandi supply in kilograms."
    )
    gap_pct: float = Field(
        description="Supply-demand gap as a percentage of predicted supply."
    )
    opportunity_score: float = Field(
        description="Opportunity score 0–100 representing market procurement urgency."
    )
    summary_headline: str = Field(
        description="One-sentence high-impact headline for the forecast."
    )
    detailed_narrative: str = Field(
        description=(
            "Natural language narrative (2-3 sentences) explaining how weather, "
            "festivals, seasonality, and live orders shaped this forecast."
        )
    )
    drivers: List[SignalDriver] = Field(
        description=(
            "List of exactly 3-5 key signal drivers. MUST always be non-empty – "
            "even under normal/baseline conditions, list standard seasonal baseline, "
            "weather window, and live order inflow as explicit drivers."
        )
    )
    farmer_advisory: str = Field(
        description="Actionable, practical recommendation for regional FPOs and farmers."
    )


# backward-compat alias used by api/main.py
AIDemandReport = DemandAdvisorySchema


# ------------------------------------------------------------------ #
# Synthesizer
# ------------------------------------------------------------------ #

class LangChainDemandSynthesizer:
    """
    Curates all market signals into a structured DemandAdvisorySchema via
    LangChain → Gemini → PydanticOutputParser.
    """

    _PROMPT_TEXT = """\
You are an expert Agricultural AI Market Intelligence Specialist for 'Kisan Guard'.
Synthesize the exact numerical market inputs below into a rigorous demand forecast report.

=== EXACT MARKET INPUTS (do NOT modify these numbers) ===
• Commodity           : {commodity}
• Target Location     : {location_name}
• Forecast Window     : {forecast_days} days
• ML Baseline Demand  : {baseline_ml_kg:,.0f} kg
• Live Platform Orders: {live_orders_kg:,.0f} kg
• Total Dynamic Demand: {final_demand_kg:,.0f} kg
• Predicted Supply    : {predicted_supply_kg:,.0f} kg
• Supply-Demand Gap   : {gap_pct:+.2f}%
• Opportunity Score   : {opportunity_score:.1f} / 100
• Weather Condition   : {weather_desc} ({temp_c}°C)
• Weather Risk Factor : {weather_risk}
• Festival Signal     : {fest_name} (multiplier: {fest_mult:+.2f})
• Seasonal Multiplier : {season_mult:.3f}
• Output Language     : {language_full}

=== STRICT INSTRUCTIONS ===
1. Use the exact numbers above verbatim in your narrative – do NOT round or invent values.
2. The `predicted_demand_kg` field MUST equal {final_demand_kg:.0f}.
3. The `predicted_supply_kg` field MUST equal {predicted_supply_kg:.0f}.
4. The `gap_pct` field MUST equal {gap_pct:.2f}.
5. The `opportunity_score` field MUST equal {opportunity_score:.1f}.
6. The `drivers` list MUST contain exactly 3-5 entries.
   *** CRITICAL: Even if weather is clear and no festival is active, you MUST still list:
       - "Standard Seasonal Baseline" (NEUTRAL / INCREASE)
       - "Normal Weather Window (No Transport Disruption)" (NEUTRAL)
       - "Active Market Order Inflow" (INCREASE if live_orders_kg > 0, else NEUTRAL)
   Add any additional active signals (festival, heat, rain) on top. ***
7. Write the entire output in {language_full}.

{format_instructions}
"""

    @classmethod
    def synthesize_report(
        cls,
        commodity: str,
        location_name: str,
        forecast_days: int,
        baseline_ml_kg: float,
        live_app_orders_kg: float,
        weather_info: Dict[str, Any],
        festival_info: Dict[str, Any] | None = None,
        season_mult: float = 1.0,
        predicted_supply_kg: float | None = None,
        language: str = "en",
    ) -> DemandAdvisorySchema:
        """
        Main entry-point called by api/main.py.

        Parameters
        ----------
        commodity        : e.g. 'Onion'
        location_name    : resolved city name from LocationEngine
        forecast_days    : int 1-30
        baseline_ml_kg   : historical ML or dynamic baseline
        live_app_orders_kg : accumulated platform orders from DemandSignalStore
        weather_info     : dict from WeatherEngine.get_live_weather()
        festival_info    : dict with 'name' and 'multiplier' keys (optional)
        season_mult      : float seasonal multiplier (default 1.0)
        predicted_supply_kg : pre-computed supply (if None, computed internally)
        language         : 'hi' for Hindi, 'en' for English
        """
        # --- Compute derived metrics ---------------------------------------------------
        fest = festival_info or {"name": "None (Standard Period)", "multiplier": 0.0}
        fest_mult = float(fest.get("multiplier", 0.0))
        weather_risk = float(weather_info.get("weather_adjustment_factor", 0.0))

        # Festival and weather impact on demand
        demand_multiplier = (1.0 + fest_mult) * season_mult
        final_demand_kg = round((baseline_ml_kg * demand_multiplier) + live_app_orders_kg, 1)

        if predicted_supply_kg is None or predicted_supply_kg <= 0:
            # Import here to avoid circular dependency
            from src.predictor import compute_predicted_supply_kg
            predicted_supply_kg = compute_predicted_supply_kg(
                commodity=commodity,
                district=location_name,
                baseline_demand_kg=baseline_ml_kg,
                weather_adjustment_factor=weather_risk,
            )

        predicted_supply_kg = round(predicted_supply_kg, 1)
        gap_kg = final_demand_kg - predicted_supply_kg
        gap_pct = round(
            (gap_kg / predicted_supply_kg) * 100.0 if predicted_supply_kg > 0 else 0.0, 2
        )

        # Live order ratio (normalized by baseline; cap at 1.0 for formula stability)
        live_order_ratio = min(1.0, live_app_orders_kg / max(1.0, baseline_ml_kg))

        opportunity_score = round(
            min(100.0, max(0.0,
                (gap_pct * 1.5)
                + (live_order_ratio * 25.0)
                + (abs(weather_risk) * 15.0)
            )),
            1,
        )

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        language_full = "Pure Hindi (Devanagari script)" if language == "hi" else "English"

        context = {
            "commodity":          commodity,
            "location_name":      location_name,
            "forecast_days":      forecast_days,
            "baseline_ml_kg":     baseline_ml_kg,
            "live_orders_kg":     live_app_orders_kg,
            "final_demand_kg":    final_demand_kg,
            "predicted_supply_kg": predicted_supply_kg,
            "gap_pct":            gap_pct,
            "opportunity_score":  opportunity_score,
            "weather_desc":       weather_info.get("condition", "Normal"),
            "temp_c":             weather_info.get("temperature_c", 25.0),
            "weather_risk":       weather_risk,
            "fest_name":          fest.get("name", "None (Standard Period)"),
            "fest_mult":          fest_mult,
            "season_mult":        season_mult,
            "language_full":      language_full,
        }

        if not api_key:
            logger.warning("GOOGLE_API_KEY / GEMINI_API_KEY missing. Using dynamic fallback report.")
            return cls._build_fallback_report(context)

        try:
            return cls._invoke_llm(context, api_key)
        except Exception as exc:
            logger.error("LangChain Gemini synthesis failed: %s", exc, exc_info=True)
            return cls._build_fallback_report(context)

    # ------------------------------------------------------------------ #
    # LLM invocation
    # ------------------------------------------------------------------ #

    @classmethod
    def _invoke_llm(cls, ctx: Dict[str, Any], api_key: str) -> DemandAdvisorySchema:
        parser = PydanticOutputParser(pydantic_object=DemandAdvisorySchema)
        prompt = PromptTemplate(
            template=cls._PROMPT_TEXT,
            input_variables=list(ctx.keys()),
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.15,
        )
        chain = prompt | llm | parser
        result: DemandAdvisorySchema = chain.invoke(ctx)

        # Enforce that numeric pins match our pre-computed values
        result.predicted_demand_kg = ctx["final_demand_kg"]
        result.predicted_supply_kg = ctx["predicted_supply_kg"]
        result.gap_pct = ctx["gap_pct"]
        result.opportunity_score = ctx["opportunity_score"]
        return result

    # ------------------------------------------------------------------ #
    # Dynamic fallback (no LLM call)
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_fallback_report(cls, ctx: Dict[str, Any]) -> DemandAdvisorySchema:
        """
        Constructs a fully dynamic advisory using all pre-computed values.
        No hardcoded numbers – every field reflects the real inputs.
        """
        commodity = ctx["commodity"]
        location  = ctx["location_name"]
        days      = ctx["forecast_days"]
        demand_kg = ctx["final_demand_kg"]
        supply_kg = ctx["predicted_supply_kg"]
        gap_pct   = ctx["gap_pct"]
        opp_score = ctx["opportunity_score"]
        live_kg   = ctx["live_orders_kg"]
        baseline  = ctx["baseline_ml_kg"]
        weather   = ctx["weather_desc"]
        temp      = ctx["temp_c"]
        fest_name = ctx["fest_name"]
        fest_mult = ctx["fest_mult"]
        season    = ctx["season_mult"]
        lang      = ctx["language_full"]

        # --- Build explicit drivers ------------------------------------------
        drivers: List[SignalDriver] = [
            SignalDriver(
                driver_name="Standard Seasonal Baseline",
                impact_direction="INCREASE" if season > 1.0 else "NEUTRAL",
                impact_magnitude_pct=round((season - 1.0) * 100.0, 1),
                explanation=(
                    f"Monthly seasonal demand model (multiplier {season:.3f}) "
                    f"applied to the {commodity} baseline of {baseline:,.0f} kg."
                ),
            ),
            SignalDriver(
                driver_name=f"Normal Weather Window ({weather})" if weather in ("Clear / Favorable", "Normal")
                            else f"Weather Impact: {weather}",
                impact_direction="NEUTRAL" if ctx["weather_risk"] == 0.0
                                else ("DECREASE" if ctx["weather_risk"] < 0 else "INCREASE"),
                impact_magnitude_pct=round(ctx["weather_risk"] * 100.0, 1),
                explanation=(
                    f"Current temperature {temp}°C with condition '{weather}'. "
                    f"Weather risk coefficient: {ctx['weather_risk']:+.2f}."
                ),
            ),
            SignalDriver(
                driver_name="Active Market Order Inflow",
                impact_direction="INCREASE" if live_kg > 0 else "NEUTRAL",
                impact_magnitude_pct=round((live_kg / max(1.0, baseline)) * 100.0, 1),
                explanation=(
                    f"{live_kg:,.0f} kg of verified platform orders accumulated "
                    f"for {commodity} in {location}, added to the ML baseline."
                ),
            ),
        ]

        # Add festival driver only when active
        if fest_name and fest_name not in ("None (Standard Period)", ""):
            drivers.append(
                SignalDriver(
                    driver_name=f"Festival Signal: {fest_name}",
                    impact_direction="INCREASE" if fest_mult > 0 else "DECREASE",
                    impact_magnitude_pct=round(fest_mult * 100.0, 1),
                    explanation=(
                        f"Festival calendar detection triggered a {fest_mult:+.2f} multiplier "
                        f"for {commodity} demand during '{fest_name}'."
                    ),
                )
            )

        # Add gap alert driver if gap > 10%
        if abs(gap_pct) > 10.0:
            drivers.append(
                SignalDriver(
                    driver_name="Supply-Demand Gap Alert",
                    impact_direction="INCREASE" if gap_pct > 0 else "DECREASE",
                    impact_magnitude_pct=round(gap_pct, 1),
                    explanation=(
                        f"Predicted mandi supply ({supply_kg:,.0f} kg) is {gap_pct:+.1f}% "
                        f"{'below' if gap_pct > 0 else 'above'} total dynamic demand."
                    ),
                )
            )

        # --- Headline & narrative --------------------------------------------
        direction_word = "surplus" if gap_pct < 0 else "deficit"
        gap_abs = abs(gap_pct)

        if lang.startswith("Pure Hindi"):
            headline = (
                f"AI मॉडल: {location} में {commodity} की मांग अगले {days} दिनों में "
                f"{demand_kg:,.0f} किग्रा अनुमानित – {gap_abs:.1f}% {direction_word}।"
            )
            narrative = (
                f"ऐतिहासिक ML आधार ({baseline:,.0f} किग्रा) और {live_kg:,.0f} किग्रा "
                f"के लाइव प्लेटफॉर्म ऑर्डर के संयोजन से कुल मांग {demand_kg:,.0f} किग्रा बनती है। "
                f"मंडी में अनुमानित आपूर्ति {supply_kg:,.0f} किग्रा है, जिससे "
                f"{gap_pct:+.1f}% का अंतर बनता है। मौसम की स्थिति ({weather}, {temp}°C) "
                f"और {fest_name} के संकेत को ध्यान में रखा गया है।"
            )
            advisory = (
                f"{location} क्षेत्र के किसान भाइयों और FPO को सलाह है कि "
                f"{commodity} की कटाई और मंडी में आवक को {days} दिनों के भीतर सुनिश्चित करें। "
                f"अवसर स्कोर {opp_score:.0f}/100 है।"
            )
        else:
            headline = (
                f"AI Forecast: {commodity} demand in {location} projected at {demand_kg:,.0f} kg "
                f"over {days} days — {gap_abs:.1f}% supply {direction_word} detected."
            )
            narrative = (
                f"The historical ML baseline of {baseline:,.0f} kg combined with {live_kg:,.0f} kg "
                f"in active platform orders yields a total dynamic demand of {demand_kg:,.0f} kg. "
                f"Predicted mandi supply stands at {supply_kg:,.0f} kg, producing a {gap_pct:+.1f}% gap. "
                f"Conditions: {weather} at {temp}°C; festival signal '{fest_name}' (x{(1+fest_mult):.2f})."
            )
            advisory = (
                f"Regional FPOs and farmers within 80 km of {location} should target dispatching "
                f"{commodity} within the {days}-day window to capture peak demand. "
                f"Opportunity Score: {opp_score:.0f}/100."
            )

        return DemandAdvisorySchema(
            predicted_demand_kg=demand_kg,
            predicted_supply_kg=supply_kg,
            gap_pct=gap_pct,
            opportunity_score=opp_score,
            summary_headline=headline,
            detailed_narrative=narrative,
            drivers=drivers,
            farmer_advisory=advisory,
        )