import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64
from pathlib import Path
from datetime import datetime, timezone

from core.physics import SpaceObject, PhysicsEngine
from core.data_ingestion import parse_tle_lines
from core.conjunction import find_tca
from core.bayesian_risk import calculate_collision_probability
from core.decision_engine import evaluate_risk, DecisionState
from core.path_planner import AStarPathPlanner
from core.edge_triage import OnboardSensorFrame, EdgeAITriageEngine

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ORBITAL // To The Unknown",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load base64 planet image for seamless offline, crisp high-res rendering
assets_dir = Path(__file__).resolve().parent / "assets"
mars_img_path = assets_dir / "mars_planet.jpg"
if mars_img_path.exists():
    with open(mars_img_path, "rb") as f:
        planet_b64 = base64.b64encode(f.read()).decode("utf-8")
        planet_src = f"data:image/jpeg;base64,{planet_b64}"
else:
    planet_src = "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=1200&q=80"

# ---------------------------------------------------------------------------
# Editorial Poster Sci-Fi CSS Styling (Matching Slider Revolution / Mercury / Venus Posters)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,900;1,400&family=Space+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* Main Canvas: Pitch black with starfield atmosphere */
.stApp {{
    background-color: #030407;
    background-image: 
        radial-gradient(1.5px 1.5px at 15% 20%, rgba(255,255,255,0.7), transparent),
        radial-gradient(1px 1px at 45% 15%, rgba(255,255,255,0.5), transparent),
        radial-gradient(1.5px 1.5px at 75% 35%, rgba(255,255,255,0.8), transparent),
        radial-gradient(2px 2px at 90% 70%, rgba(255,255,255,0.6), transparent),
        radial-gradient(1px 1px at 30% 80%, rgba(255,255,255,0.4), transparent),
        radial-gradient(circle at 65% 45%, #180e12 0%, #08070d 50%, #030407 100%);
    color: #cbd5e1;
    font-family: 'Space Grotesk', -apple-system, sans-serif;
}}

header {{visibility: hidden;}}
footer {{visibility: hidden;}}
.block-container {{
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 96rem;
}}

/* Top Minimal Navigation Bar */
.poster-nav {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 30px;
}}
.nav-badge {{
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #ffffff;
}}
.nav-center-tag {{
    font-size: 11px;
    letter-spacing: 3px;
    color: #64748b;
    text-transform: uppercase;
}}
.nav-links {{
    display: flex;
    gap: 36px;
    font-size: 12px;
    letter-spacing: 3px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 500;
}}
.nav-links a {{
    color: #94a3b8;
    text-decoration: none;
    transition: color 0.2s;
}}
.nav-links a:hover {{
    color: #ffffff;
}}

/* Giant Poster Typography */
.poster-hero {{
    position: relative;
    padding: 20px 0 10px 0;
}}
.poster-number {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 4px;
    color: #f97316;
    margin-bottom: 12px;
    text-transform: uppercase;
}}
.poster-title {{
    font-family: 'Playfair Display', serif;
    font-size: 88px;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -2px;
    color: #ffffff;
    margin-bottom: 18px;
    text-shadow: 0 15px 40px rgba(0,0,0,0.8);
}}
.poster-title span {{
    font-weight: 400;
    font-style: italic;
    color: #e2e8f0;
}}
.poster-subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 4px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 24px;
}}
.poster-body {{
    font-size: 15px;
    line-height: 1.8;
    color: #94a3b8;
    max-width: 480px;
    margin-bottom: 30px;
}}

/* The Planet Art Container with Glow */
.planet-stage {{
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.planet-img {{
    width: 100%;
    max-width: 580px;
    border-radius: 50%;
    box-shadow: 
        -25px 0 80px rgba(249, 115, 22, 0.25),
        20px 0 90px rgba(2, 132, 199, 0.2),
        inset -20px 0 60px rgba(0,0,0,0.8);
    filter: drop-shadow(0 20px 50px rgba(0,0,0,0.9));
    animation: slow-drift 60s ease-in-out infinite alternate;
}}
@keyframes slow-drift {{
    0% {{ transform: rotate(0deg) scale(1.0); }}
    100% {{ transform: rotate(8deg) scale(1.03); }}
}}

/* Status Indicator Pill */
.alert-tag {{
    display: inline-block;
    padding: 8px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 4px;
    margin-bottom: 24px;
}}
.alert-red {{
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid #ef4444;
    color: #f87171;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
}}
.alert-amber {{
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
}}
.alert-emerald {{
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10b981;
    color: #34d399;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
}}

/* Editorial Metadata Grid (Like the bottom stats on Mercury/Venus posters) */
.stat-bar {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    padding: 24px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin: 35px 0;
}}
.stat-cell {{
    border-left: 1px solid rgba(255, 255, 255, 0.08);
    padding-left: 16px;
}}
.stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.stat-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.5px;
}}
.stat-unit {{
    font-size: 14px;
    font-weight: 400;
    color: #94a3b8;
}}
.stat-sub {{
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
}}
</style>

<!-- Top Minimal Navigation -->
<div class="poster-nav">
    <div class="nav-badge">01 - 03</div>
    <div class="nav-center-tag">AUTONOMOUS ORBITAL DEFLECTION PLATFORM</div>
    <div class="nav-links">
        <a href="#conjunction">_CONJUNCTION</a>
        <a href="#trajectory">_TRAJECTORY</a>
        <a href="#triage">_EDGE AI</a>
        <a href="#telemetry">_METRICS</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Mission Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<p style='font-family: JetBrains Mono; font-size: 12px; letter-spacing: 3px; color: #f97316;'>// MISSION PARAMETERS</p>", unsafe_allow_html=True)
    
    scenario = st.selectbox(
        "ENCOUNTER PRESET",
        ["ISS vs Cosmos-2251 Fragment (Critical LEO)", "CSS Tianhe vs Fengyun Debris"]
    )
    sensor_noise = st.slider("TRACKING UNCERTAINTY (σ km)", 0.05, 1.5, 0.20, step=0.05)
    remaining_fuel = st.slider("AVAILABLE ΔV THRUST UNITS", 0.0, 100.0, 85.0, step=5.0)
    enable_triage = st.checkbox("RUN ONBOARD EDGE AI TRIAGE", value=True)

# ---------------------------------------------------------------------------
# Core Astrodynamics & Conjunction Calculation
# ---------------------------------------------------------------------------
now_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
miss_distance_km = 0.38
rel_vel_km_s = 14.28

# Bayesian Risk
p_coll = calculate_collision_probability(
    miss_distance_km=miss_distance_km,
    sigma_km=sensor_noise,
    hard_body_radius_m=6.5
)

# Decision Engine
decision = evaluate_risk(
    p_collision=p_coll,
    miss_distance_km=miss_distance_km,
    fuel_remaining_delta_v=remaining_fuel
)

# A* Path Planner
planner = AStarPathPlanner(step_size=0.4, safety_margin_km=1.2)
sat_pos = np.array([0.0, 0.0, 0.0])
deb_pos = np.array([2.5, 0.25, 0.0])
target_pos = np.array([5.5, 0.0, 0.0])
evasive_path = planner.plan_evasive_path(sat_pos, target_pos, [deb_pos])
delta_v_cost = planner.calculate_path_delta_v(evasive_path)

# ---------------------------------------------------------------------------
# Hero Poster Layout: Giant Typography Over Planet Art
# ---------------------------------------------------------------------------
poster_col1, poster_col2 = st.columns([1.15, 1.0], gap="large")

with poster_col1:
    st.markdown('<div class="poster-number">ORBITAL REGIME // LEO 415 KM</div>', unsafe_allow_html=True)
    st.markdown('<div class="poster-title">to the<br><span>unknown</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="poster-subtitle">AND SAFE PASSAGE HOME</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="poster-body">'
        'High-velocity orbital debris fragments represent a critical existential threat to crewed and autonomous '
        'spacecraft. This autonomous platform integrates <b>SGP4 astrodynamics</b>, <b>Bayesian conjunction screening</b>, '
        'and <b>A* kinetic maneuvering</b> to guarantee orbital survival.'
        '</div>',
        unsafe_allow_html=True
    )

    # Decision alert tag
    if decision in ["EMERGENCY_AVOIDANCE", "CRITICAL_NO_FUEL_ALERT"]:
        st.markdown(f'<div class="alert-tag alert-red">● DEFLECTION REQUIRED // {decision}</div>', unsafe_allow_html=True)
    elif decision == "PLAN_EVASIVE_MANEUVER":
        st.markdown(f'<div class="alert-tag alert-amber">● MONITOR CONJUNCTION // {decision}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-tag alert-emerald">● TRAJECTORY NOMINAL // {decision}</div>', unsafe_allow_html=True)

with poster_col2:
    # Editorial Planet Visual with Atmospheric Depth
    st.markdown(f"""
    <div class="planet-stage">
        <img class="planet-img" src="{planet_src}" alt="Celestial Planet Body">
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Editorial Metadata Grid (Mercury / Venus Bottom Bar Style)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="stat-bar">
    <div class="stat-cell">
        <div class="stat-label">COLLISION PROBABILITY</div>
        <div class="stat-val" style="color: #f43f5e;">{p_coll:.2e}</div>
        <div class="stat-sub">R = 6.5m Hard-Sphere B-Plane</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">MISS DISTANCE AT TCA</div>
        <div class="stat-val" style="color: #38bdf8;">{miss_distance_km:.2f} <span class="stat-unit">KM</span></div>
        <div class="stat-sub">Rel Velocity: {rel_vel_km_s:.1f} km/s</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">AVOIDANCE THRUST (ΔV)</div>
        <div class="stat-val" style="color: #10b981;">{delta_v_cost:.1f} <span class="stat-unit">M/S</span></div>
        <div class="stat-sub">3D A* Optimal Maneuver</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">RESERVE PROPELLANT</div>
        <div class="stat-val" style="color: #fbbf24;">{remaining_fuel:.0f} <span class="stat-unit">ΔV</span></div>
        <div class="stat-sub">Hydrazine RCS Reserves</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3D Interactive Spatial Encounter & A* Trajectory Detour
# ---------------------------------------------------------------------------
st.markdown('<div class="poster-number" id="trajectory">// 02 SPATIAL TRAJECTORY VISUALIZATION</div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Playfair Display; font-size: 38px; color: #ffffff; margin-top: 0;">Kinetic Conjunction & Detour</h2>', unsafe_allow_html=True)

fig = go.Figure()

# 1. Primary Satellite
fig.add_trace(go.Scatter3d(
    x=[sat_pos[0]], y=[sat_pos[1]], z=[sat_pos[2]],
    mode='markers+text',
    marker=dict(size=11, color='#38bdf8', symbol='circle', line=dict(color='#ffffff', width=2)),
    text=['ISS (PRIMARY SATELLITE)'],
    textfont=dict(family='JetBrains Mono', size=11, color='#38bdf8'),
    textposition='top center',
    name='Primary Spacecraft'
))

# 2. Debris Obstacle
fig.add_trace(go.Scatter3d(
    x=[deb_pos[0]], y=[deb_pos[1]], z=[deb_pos[2]],
    mode='markers+text',
    marker=dict(size=9, color='#ef4444', symbol='diamond', line=dict(color='#fca5a5', width=1.5)),
    text=['COSMOS-2251 DEBRIS'],
    textfont=dict(family='JetBrains Mono', size=11, color='#ef4444'),
    textposition='bottom center',
    name='Debris Fragment'
))

# 3. Translucent Keep-Out Safety Bubble
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
r_h = 1.0
hx = deb_pos[0] + r_h * np.outer(np.cos(u), np.sin(v))
hy = deb_pos[1] + r_h * np.outer(np.sin(u), np.sin(v))
hz = deb_pos[2] + r_h * np.outer(np.ones(np.size(u)), np.cos(v))

fig.add_trace(go.Surface(
    x=hx, y=hy, z=hz,
    colorscale=[[0, 'rgba(239, 68, 68, 0.12)'], [1, 'rgba(239, 68, 68, 0.3)']],
    showscale=False,
    name='Keep-Out Sphere'
))

# 4. A* Evasive Detour Waypoints
path_np = np.array(evasive_path)
fig.add_trace(go.Scatter3d(
    x=path_np[:, 0], y=path_np[:, 1], z=path_np[:, 2],
    mode='lines+markers',
    line=dict(color='#10b981', width=6),
    marker=dict(size=4, color='#34d399'),
    name='A* Evasive Path'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(0,0,0,0)', title='X (km)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(0,0,0,0)', title='Y (km)'),
        zaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(0,0,0,0)', title='Z (km)'),
        bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.4, y=-1.5, z=1.0))
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='JetBrains Mono'),
    margin=dict(l=0, r=0, b=0, t=20),
    height=480,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Onboard Edge AI Data Triage Section
# ---------------------------------------------------------------------------
if enable_triage:
    st.markdown('<div class="poster-number" id="triage">// 03 EDGE INTELLIGENCE LAYER</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family: Playfair Display; font-size: 38px; color: #ffffff; margin-top: 0;">Onboard Data Triage</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 14px; margin-bottom: 25px;">Lightweight FPGA/Edge classifier filtering cloud-obscured and blank frames in space, conserving downlink bandwidth.</p>', unsafe_allow_html=True)

    triage_engine = EdgeAITriageEngine()
    test_frames = [
        OnboardSensorFrame("RAW_OPT_01", cloud_cover_pct=85.0, signal_to_noise_ratio=14.0, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("RAW_OPT_02", cloud_cover_pct=15.0, signal_to_noise_ratio=26.0, debris_streak_detected=True, raw_size_mb=45.0),
        OnboardSensorFrame("RAW_OPT_03", cloud_cover_pct=92.0, signal_to_noise_ratio=08.0, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("RAW_OPT_04", cloud_cover_pct=04.0, signal_to_noise_ratio=0.9, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("RAW_OPT_05", cloud_cover_pct=24.0, signal_to_noise_ratio=18.0, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("RAW_OPT_06", cloud_cover_pct=79.0, signal_to_noise_ratio=12.0, debris_streak_detected=False, raw_size_mb=45.0),
    ]

    for f in test_frames:
        triage_engine.classify_frame(f)

    rep = triage_engine.get_savings_report()

    st.markdown(f"""
    <div class="stat-bar" style="margin-top: 10px;">
        <div class="stat-cell">
            <div class="stat-label">FRAMES EVALUATED</div>
            <div class="stat-val">{rep['total_frames_processed']}</div>
            <div class="stat-sub">Onboard FPGA Pipeline</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">RAW OPTICAL MEMORY</div>
            <div class="stat-val">{rep['total_raw_data_mb']} <span class="stat-unit">MB</span></div>
            <div class="stat-sub">Unfiltered Sensor Memory</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">GROUND DOWNLINK</div>
            <div class="stat-val" style="color: #38bdf8;">{rep['actual_downlinked_data_mb']} <span class="stat-unit">MB</span></div>
            <div class="stat-sub">Priority Telemetry Queue</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">BANDWIDTH CONSERVED</div>
            <div class="stat-val" style="color: #10b981;">{rep['bandwidth_saved_pct']}</div>
            <div class="stat-sub">{rep['bandwidth_saved_mb']} MB Filtered Onboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
