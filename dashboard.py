# dashboard.py
"""
Kisan Guard – Streamlit Intelligence Dashboard  v5.0
=====================================================

Architecture
------------
- All demand/supply/gap/score numbers come DIRECTLY from the FastAPI
  backend (/api/v1/predict-demand).  The dashboard does NOT recompute
  them locally so there is a single source of truth.
- Logging an order via /api/v1/log-order immediately calls st.rerun()
  which re-invokes the forecast endpoint, picking up the new order from
  the in-memory DemandSignalStore.
- Session state tracks the last successful forecast result so charts
  persist between order-log events without a full re-render lag.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------------ #
# Page config & CSS
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Kisan Guard | AI Ag-Demand Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #050C18; color: #E2E8F0; }

/* Cards */
.kgcard {
    background: rgba(15,23,42,0.82);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    text-align: center;
    transition: transform .15s ease;
}
.kgcard:hover { transform: translateY(-2px); }
.kg-label { color: #64748B; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
.kg-value { color: #F8FAFC; font-size: 1.85rem; font-weight: 800; margin-top: 6px; line-height: 1; }
.kg-sub   { color: #94A3B8; font-size: 0.78rem; margin-top: 4px; }

/* Alert cards */
.alert-good  { border-left: 4px solid #10B981; }
.alert-warn  { border-left: 4px solid #F59E0B; }
.alert-danger{ border-left: 4px solid #EF4444; }

/* Advisory block */
.adv-card {
    background: rgba(15,23,42,0.88);
    border-left: 4px solid #3B82F6;
    padding: 22px 26px;
    border-radius: 10px;
    margin-bottom: 18px;
}
.adv-headline { color: #60A5FA; font-size: 1.1rem; font-weight: 700; margin: 0 0 10px; }

/* Driver pills */
.drv-pill-up   { display:inline-block; background:rgba(16,185,129,.15); color:#6EE7B7; border:1px solid rgba(16,185,129,.3); padding:3px 10px; border-radius:6px; font-size:.78rem; margin:3px 4px 3px 0; }
.drv-pill-down { display:inline-block; background:rgba(239,68,68,.15);  color:#FCA5A5; border:1px solid rgba(239,68,68,.3);  padding:3px 10px; border-radius:6px; font-size:.78rem; margin:3px 4px 3px 0; }
.drv-pill-neu  { display:inline-block; background:rgba(148,163,184,.12); color:#CBD5E1; border:1px solid rgba(148,163,184,.25); padding:3px 10px; border-radius:6px; font-size:.78rem; margin:3px 4px 3px 0; }

/* Sidebar niceties */
section[data-testid="stSidebar"] > div { background: #080E1A; }

/* Score gauge */
.score-ring {
    width: 110px; height: 110px; border-radius: 50%; margin: 0 auto 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# ------------------------------------------------------------------ #
# Session state initialization
# ------------------------------------------------------------------ #

def _init_state() -> None:
    defaults = {
        "user_lat":       28.5355,
        "user_lon":       77.3910,
        "detected_city":  "Noida",
        "detected_state": "Uttar Pradesh",
        "last_forecast":  None,   # holds last successful /predict-demand response
        "order_log":      [],     # local display-only order history
        "forecast_run":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ------------------------------------------------------------------ #
# API helpers
# ------------------------------------------------------------------ #

def _api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _run_forecast(
    commodity: str,
    location_query: str,
    lat: float,
    lon: float,
    forecast_days: int,
    language: str,
) -> Optional[Dict[str, Any]]:
    """POST to /api/v1/predict-demand and return the parsed JSON."""
    payload = {
        "commodity":      commodity,
        "location_query": location_query,
        "latitude":       lat,
        "longitude":      lon,
        "forecast_days":  forecast_days,
        "language":       "hi" if language == "Hindi" else "en",
    }
    try:
        r = requests.post(f"{API_BASE}/api/v1/predict-demand", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"API Error {r.status_code}: {r.text[:300]}")
    except Exception as exc:
        st.error(f"Backend connection failed: {exc}")
    return None


def _post_order(
    commodity: str,
    location: str,
    quantity_kg: float,
    buyer_type: str,
) -> Optional[Dict[str, Any]]:
    """POST to /api/v1/log-order."""
    payload = {
        "commodity":   commodity,
        "location":    location,
        "quantity_kg": quantity_kg,
        "buyer_type":  buyer_type.lower(),
    }
    try:
        r = requests.post(f"{API_BASE}/api/v1/log-order", json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()
        st.error(f"Order POST failed {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        st.error(f"Order submission error: {exc}")
    return None


def _fetch_orders(commodity: str, location: str) -> List[Dict[str, Any]]:
    try:
        r = requests.get(
            f"{API_BASE}/orders",
            params={"commodity": commodity, "location": location},
            timeout=2.5,
        )
        if r.status_code == 200:
            return r.json() if isinstance(r.json(), list) else []
    except Exception:
        pass
    return []

# ------------------------------------------------------------------ #
# Backend gate
# ------------------------------------------------------------------ #

if not _api_health():
    st.error(
        "🚨 **FastAPI Backend is Offline**\n\n"
        "Start the server with:\n```\nuvicorn api.main:app --reload\n```"
    )
    st.stop()

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #

st.sidebar.markdown("## 📍 Location & Market Setup")

# --- IP geolocation ---
if st.sidebar.button("📡 Auto-Detect My Location (IP)", use_container_width=True):
    with st.sidebar:
        with st.spinner("Detecting via IP…"):
            try:
                ip_r = requests.get("https://ipapi.co/json/", timeout=3.0).json()
                st.session_state.user_lat       = float(ip_r.get("latitude", 28.5355))
                st.session_state.user_lon       = float(ip_r.get("longitude", 77.3910))
                st.session_state.detected_city  = ip_r.get("city", "Noida")
                st.session_state.detected_state = ip_r.get("region", "Uttar Pradesh")
                st.success(f"Located: {st.session_state.detected_city}, {st.session_state.detected_state}")
            except Exception:
                st.warning("IP geolocation unavailable.")

# --- Manual override ---
sb_c1, sb_c2 = st.sidebar.columns(2)
with sb_c1:
    user_state = st.text_input("State", value=st.session_state.detected_state, key="sb_state")
with sb_c2:
    user_city = st.text_input("District / City", value=st.session_state.detected_city, key="sb_city")

selected_crop = st.sidebar.selectbox(
    "🌿 Target Commodity",
    ["Onion", "Potato", "Tomato", "Wheat"],
    key="sb_crop",
)
forecast_days = st.sidebar.slider("📅 Forecast Window (days)", 1, 30, 7, key="sb_days")
language      = st.sidebar.radio("🗣️ Advisory Language", ["English", "Hindi"], key="sb_lang")

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Active Signal Feed**")
now_ts = datetime.now()
st.sidebar.caption(f"🕐 System Time: `{now_ts.strftime('%d %b %Y, %H:%M')}`")
st.sidebar.caption(f"📍 Coords: `{st.session_state.user_lat:.4f}, {st.session_state.user_lon:.4f}`")

# ------------------------------------------------------------------ #
# Main header
# ------------------------------------------------------------------ #

st.markdown("# 🌾 Kisan Guard")
st.markdown("**AI Ag-Market Intelligence** · Live Order Stream · LightGBM · LangChain Gemini")

hc1, hc2, hc3 = st.columns([2, 2, 2])
with hc1:
    st.markdown(f"**Target Market:** `{user_city}, {user_state}`")
with hc2:
    st.markdown(f"**Commodity:** `{selected_crop}` · **Window:** `{forecast_days} days`")
with hc3:
    run_btn = st.button(
        "🚀 Run Demand Forecast",
        type="primary",
        use_container_width=True,
        key="btn_run_forecast",
    )

st.markdown("---")

# ------------------------------------------------------------------ #
# Run forecast
# ------------------------------------------------------------------ #

if run_btn:
    location_query = f"{user_city}, {user_state}"
    with st.spinner(f"🔄 Running AI forecast for **{selected_crop}** in **{location_query}**…"):
        result = _run_forecast(
            commodity=selected_crop,
            location_query=location_query,
            lat=st.session_state.user_lat,
            lon=st.session_state.user_lon,
            forecast_days=forecast_days,
            language=language,
        )
    if result:
        st.session_state.last_forecast = result
        st.session_state.forecast_run  = True

# ------------------------------------------------------------------ #
# Results display
# ------------------------------------------------------------------ #

if st.session_state.forecast_run and st.session_state.last_forecast:
    res: Dict[str, Any]    = st.session_state.last_forecast
    report: Dict[str, Any] = res.get("ai_report", {})
    drivers: List[Dict]    = report.get("drivers", [])

    # Extract key metrics from API response (all dynamically computed server-side)
    baseline_kg      = float(res.get("baseline_ml_kg",      0.0))
    live_orders_kg   = float(res.get("live_app_orders_kg",  0.0))
    final_demand_kg  = float(res.get("final_demand_kg",     0.0))
    supply_kg        = float(res.get("predicted_supply_kg", 0.0))
    gap_kg           = float(res.get("gap_kg",              0.0))
    gap_pct          = float(res.get("gap_pct",             0.0))
    opp_score        = float(res.get("opportunity_score",   0.0))
    fest_name        = res.get("festival", {}).get("name", "None")
    fest_mult        = float(res.get("festival", {}).get("multiplier", 0.0))
    season_mult      = float(res.get("season_mult", 1.0))
    weather_desc     = res.get("weather", {}).get("condition", "Normal")
    temp_c           = float(res.get("weather", {}).get("temperature_c", 25.0))
    mapped_dist      = res.get("mapped_district", {}).get("district", "—")
    dist_km          = res.get("mapped_district", {}).get("distance_km", 0.0)

    # ---- Advisory Block ----
    headline  = report.get("summary_headline", "—")
    narrative = report.get("detailed_narrative", "—")
    farmer_adv = report.get("farmer_advisory", "—")

    st.markdown("### 📝 Dynamic Demand Advisory")
    st.markdown(f"""
    <div class="adv-card">
        <p class="adv-headline">{headline}</p>
        <p style="color:#CBD5E1; font-size:1rem; margin:0 0 12px;">{narrative}</p>
        <p style="color:#94A3B8; font-size:.88rem;"><strong>💡 Farmer Advisory:</strong> {farmer_adv}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Signal Drivers ----
    st.markdown("#### 🔍 Active Signal Drivers")
    if drivers:
        pill_html = ""
        for d in drivers:
            direction = d.get("impact_direction", "NEUTRAL").upper()
            name      = d.get("driver_name", "—")
            mag       = d.get("impact_magnitude_pct", 0.0)
            cls_name  = "drv-pill-up" if direction == "INCREASE" else ("drv-pill-down" if direction == "DECREASE" else "drv-pill-neu")
            arrow     = "↑" if direction == "INCREASE" else ("↓" if direction == "DECREASE" else "→")
            pill_html += f'<span class="{cls_name}">{arrow} {name} ({mag:+.1f}%)</span>'
        st.markdown(pill_html, unsafe_allow_html=True)
        st.markdown("")

        # Detailed driver table
        driver_table = pd.DataFrame([
            {
                "Signal Driver":      d.get("driver_name", "—"),
                "Direction":          d.get("impact_direction", "NEUTRAL"),
                "Magnitude (%)":      f"{d.get('impact_magnitude_pct', 0.0):+.1f}%",
                "Explanation":        d.get("explanation", "—"),
            }
            for d in drivers
        ])
        st.dataframe(driver_table, use_container_width=True, hide_index=True)
    else:
        st.info("No signal drivers returned by the advisory engine.")

    st.markdown("---")

    # ---- Metric Cards ----
    st.markdown("### 📈 Calculated Metrics Summary")

    # Determine alert class for gap card
    gap_cls = "alert-good" if gap_pct < 5 else ("alert-warn" if gap_pct < 15 else "alert-danger")
    score_color = "#10B981" if opp_score < 30 else ("#F59E0B" if opp_score < 65 else "#EF4444")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">ML Baseline</div>
            <div class="kg-value">{baseline_kg:,.0f}</div>
            <div class="kg-sub">kg · {forecast_days}-day window</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">Live Orders</div>
            <div class="kg-value" style="color:#60A5FA;">{live_orders_kg:,.0f}</div>
            <div class="kg-sub">kg · active platform orders</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">Total Dynamic Demand</div>
            <div class="kg-value">{final_demand_kg:,.0f}</div>
            <div class="kg-sub">kg · season×festival adjusted</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kgcard {gap_cls}">
            <div class="kg-label">Supply Deficit</div>
            <div class="kg-value" style="color:#EF4444;">{gap_pct:+.2f}%</div>
            <div class="kg-sub">{gap_kg:+,.0f} kg gap · supply {supply_kg:,.0f} kg</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">Opportunity Score</div>
            <div class="kg-value" style="color:{score_color};">{opp_score:.1f}<span style="font-size:1rem;color:#64748B;">/100</span></div>
            <div class="kg-sub">procurement urgency index</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ---- Charts ----
    chart_c1, chart_c2 = st.columns([3, 2])

    with chart_c1:
        st.subheader("📊 Signal Impact Waterfall")

        # Compute per-signal deltas (multiplicative breakdown on baseline)
        fest_delta   = baseline_kg * fest_mult
        season_delta = baseline_kg * (season_mult - 1.0)
        weather_risk = float(res.get("weather", {}).get("weather_adjustment_factor", 0.0))
        wx_delta     = baseline_kg * weather_risk

        waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["ML Baseline", "Festival Signal", "Seasonal Factor", "Weather Risk", "Live API Orders", "Final Demand"],
            y=[baseline_kg, fest_delta, season_delta, wx_delta, live_orders_kg, 0],
            connector={"line": {"color": "#334155"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#3B82F6"}},
            totals={"marker": {"color": "#10B981"}},
            text=[
                f"{baseline_kg:,.0f} kg",
                f"{fest_delta:+,.0f} kg",
                f"{season_delta:+,.0f} kg",
                f"{wx_delta:+,.0f} kg",
                f"{live_orders_kg:+,.0f} kg",
                f"{final_demand_kg:,.0f} kg",
            ],
            textposition="outside",
        ))
        waterfall.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Kilograms",
        )
        st.plotly_chart(waterfall, use_container_width=True)

    with chart_c2:
        st.subheader("🎯 Supply vs. Demand Deficit")

        supply_shown  = max(0.0, supply_kg)
        deficit_shown = max(0.0, gap_kg)

        donut = go.Figure(data=[go.Pie(
            labels=["Predicted Supply", "Unmet Demand Gap"],
            values=[supply_shown, deficit_shown] if deficit_shown > 0 else [supply_shown, 1.0],
            hole=0.62,
            marker_colors=["#10B981", "#EF4444"],
            textinfo="label+percent",
            textfont_size=11,
        )])
        donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[{
                "text":      f"{gap_pct:+.1f}%",
                "x": 0.5, "y": 0.5,
                "font_size": 22,
                "font_color": "#EF4444" if gap_pct > 0 else "#10B981",
                "showarrow": False,
            }],
            showlegend=True,
        )
        st.plotly_chart(donut, use_container_width=True)

    # ---- Signal context bar ----
    st.markdown("---")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">🌤️ Weather</div>
            <div style="font-size:1rem;font-weight:700;margin-top:8px;">{weather_desc}</div>
            <div class="kg-sub">{temp_c}°C</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">🎉 Festival Signal</div>
            <div style="font-size:.9rem;font-weight:700;margin-top:8px;">{fest_name}</div>
            <div class="kg-sub">Multiplier: {fest_mult:+.2f}</div>
        </div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">📅 Seasonal Curve</div>
            <div style="font-size:1.4rem;font-weight:700;margin-top:8px;">{season_mult:.3f}×</div>
            <div class="kg-sub">Monthly demand multiplier</div>
        </div>""", unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""
        <div class="kgcard">
            <div class="kg-label">🗺️ Mapped District</div>
            <div style="font-size:.95rem;font-weight:700;margin-top:8px;">{mapped_dist}</div>
            <div class="kg-sub">{dist_km} km from query point</div>
        </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # Order Entry Form
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown("### 🛒 Log New Buyer Order")
    st.caption("Posted directly to `/api/v1/log-order` → updates forecast demand instantly on re-run.")

    with st.form("order_form", clear_on_submit=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            ord_qty = st.number_input(
                "Quantity (kg)", min_value=100.0, max_value=100_000.0,
                value=5000.0, step=500.0,
            )
        with fc2:
            ord_location = st.text_input("Ship-to Location", value=f"{user_city}, {user_state}")
        with fc3:
            ord_buyer = st.selectbox(
                "Buyer Category",
                ["Retailer", "Wholesaler", "Institution", "Processor", "Export"],
            )
        with fc4:
            ord_crop = st.selectbox("Commodity", ["Onion", "Potato", "Tomato", "Wheat"],
                                    index=["Onion", "Potato", "Tomato", "Wheat"].index(selected_crop))

        submit_order = st.form_submit_button("📦 Post Order to Backend", use_container_width=True)
        if submit_order:
            with st.spinner("Posting order…"):
                result_order = _post_order(
                    commodity=ord_crop,
                    location=ord_location,
                    quantity_kg=ord_qty,
                    buyer_type=ord_buyer,
                )
            if result_order:
                total_logged = result_order.get("total_active_platform_orders_kg", 0.0)
                st.success(
                    f"✅ Order logged: **{ord_qty:,.0f} kg** of **{ord_crop}** from `{ord_location}`. "
                    f"Total platform orders: **{total_logged:,.0f} kg**. "
                    f"Re-run the forecast to see updated demand."
                )
                # Store in local display log
                st.session_state.order_log.append({
                    "Time":         datetime.now().strftime("%H:%M:%S"),
                    "Commodity":    ord_crop,
                    "Location":     ord_location,
                    "Qty (kg)":     f"{ord_qty:,.0f}",
                    "Buyer":        ord_buyer,
                })
                # Trigger automatic re-forecast so charts update immediately
                new_forecast = _run_forecast(
                    commodity=selected_crop,
                    location_query=f"{user_city}, {user_state}",
                    lat=st.session_state.user_lat,
                    lon=st.session_state.user_lon,
                    forecast_days=forecast_days,
                    language=language,
                )
                if new_forecast:
                    st.session_state.last_forecast = new_forecast
                st.rerun()

    # ---- Local order history ----
    if st.session_state.order_log:
        st.markdown("#### 📋 This Session's Logged Orders")
        st.dataframe(
            pd.DataFrame(st.session_state.order_log),
            use_container_width=True,
            hide_index=True,
        )

    # ---- Live order pull from API ----
    api_orders = _fetch_orders(selected_crop, user_city)
    if api_orders:
        st.markdown(f"#### 🔴 Live Backend Orders: {selected_crop} in {user_city} ({len(api_orders)} total)")
        order_df = pd.DataFrame(api_orders)[
            [c for c in ["order_id","commodity","location","quantity_kg","buyer_type","logged_at"] if c in pd.DataFrame(api_orders).columns]
        ]
        st.dataframe(order_df, use_container_width=True, hide_index=True)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-size: 5rem;">🌾</div>
        <h2 style="color:#60A5FA; margin-top:16px;">Kisan Guard Intelligence Engine</h2>
        <p style="color:#94A3B8; font-size:1.05rem; max-width:560px; margin:0 auto 28px;">
            Set your target commodity and location in the sidebar, then click
            <strong style="color:#F8FAFC;">🚀 Run Demand Forecast</strong> to generate a live AI market analysis.
        </p>
        <p style="color:#475569; font-size:.9rem;">
            Powered by LightGBM · LangChain Gemini · FastAPI · Open-Meteo
        </p>
    </div>
    """, unsafe_allow_html=True)