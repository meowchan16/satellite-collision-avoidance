import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

from core.physics import SpaceObject, PhysicsEngine
from core.data_ingestion import parse_tle_lines
from core.conjunction import find_tca
from core.bayesian_risk import calculate_collision_probability
from core.decision_engine import evaluate_risk, DecisionState
from core.path_planner import AStarPathPlanner
from core.edge_triage import OnboardSensorFrame, EdgeAITriageEngine

# ---------------------------------------------------------------------------
# Page Configuration & Immersive Dark Space Sci-Fi Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ORBITAL // Autonomous Space Situational Awareness",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep space typography, frosted glass, glowing borders, and planetary aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Cinzel:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* 1. Main Background */
.stApp {
    background-color: #0B0E14;
    background-image: radial-gradient(circle at 50% 50%, #1A2332 0%, #0B0E14 100%);
    color: #E0E0E0;
    font-family: 'Space Grotesk', -apple-system, sans-serif;
}

/* 2. Glassmorphism Metric Cards */
div.css-1r6slb0, div.stMetric, [data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
    backdrop-filter: blur(10px);
}

/* 3. Neon Orbitron Headings */
h1, h2, h3 {
    color: #00E5FF !important;
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
}

/* 4. Alert Styling (For Decision Engine) */
.alert-emergency {
    color: #FF3366 !important;
    font-weight: bold;
    border: 1px solid #FF3366 !important;
    padding: 6px 14px;
    border-radius: 6px;
    background: rgba(255, 51, 102, 0.15) !important;
    box-shadow: 0 0 12px rgba(255, 51, 102, 0.4);
    font-family: 'Orbitron', monospace;
    display: inline-block;
}
.alert-warning {
    color: #FFB800 !important;
    font-weight: bold;
    border: 1px solid #FFB800 !important;
    padding: 6px 14px;
    border-radius: 6px;
    background: rgba(255, 184, 0, 0.15) !important;
    box-shadow: 0 0 12px rgba(255, 184, 0, 0.4);
    font-family: 'Orbitron', monospace;
    display: inline-block;
}
.alert-safe {
    color: #00FF9D !important;
    font-weight: bold;
    border: 1px solid #00FF9D !important;
    padding: 6px 14px;
    border-radius: 6px;
    background: rgba(0, 255, 157, 0.15) !important;
    box-shadow: 0 0 12px rgba(0, 255, 157, 0.4);
    font-family: 'Orbitron', monospace;
    display: inline-block;
}

/* Hide standard Streamlit header & decoration */
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 95rem;
}

/* Top Navigation Bar */
.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 28px;
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 30px;
}
.brand-mark {
    font-family: 'Cinzel', serif;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 5px;
    color: #f8fafc;
    text-transform: uppercase;
}
.nav-menu {
    display: flex;
    gap: 32px;
    font-size: 13px;
    letter-spacing: 2px;
    color: #94a3b8;
    text-transform: uppercase;
}
.nav-item-active {
    color: #38bdf8 !important;
    font-weight: 600;
    border-bottom: 2px solid #38bdf8;
    padding-bottom: 4px;
}

/* Hero Section */
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #38bdf8;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 64px;
    font-weight: 700;
    letter-spacing: 4px;
    line-height: 1.05;
    background: linear-gradient(180deg, #ffffff 20%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
}
.hero-desc {
    color: #94a3b8;
    font-size: 15px;
    line-height: 1.7;
    max-width: 580px;
    margin-bottom: 24px;
}

/* Glassmorphism Cards */
.hud-card {
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 24px;
    transition: all 0.3s ease;
}
.hud-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.15);
}
.hud-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hud-value {
    font-family: 'Cinzel', serif;
    font-size: 28px;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 4px;
}
.hud-sub {
    font-size: 12px;
    color: #94a3b8;
}

/* Pulse indicator badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.pill-danger {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.pill-warn {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.pill-safe {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* Telemetry table styling */
div[data-testid="stMetricValue"] {
    font-family: 'Cinzel', serif !important;
    font-size: 26px !important;
    color: #f8fafc !important;
}
</style>

<div class="nav-container">
    <div class="brand-mark">ORBITAL // SSA</div>
    <div class="nav-menu">
        <span class="nav-item-active">01 // CONJUNCTION</span>
        <span>02 // TRAJECTORY</span>
        <span>03 // EDGE AI TRIAGE</span>
        <span>04 // TELEMETRY</span>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #38bdf8; letter-spacing: 1px;">
        SYS_STATUS: ACTIVE ●
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Settings in Sleek Sci-Fi Tone
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family: Cinzel; font-size: 18px; color: #f8fafc; letter-spacing: 2px; margin-bottom: 15px;'>// MISSION CONFIG</div>", unsafe_allow_html=True)

    target_body = st.selectbox(
        "PRIMARY CELESTIAL TARGET",
        ["Earth Orbit (LEO Conjunction)", "Mars Orbital Transit", "Lunar Gateway Encounter"]
    )

    encounter_mode = st.radio(
        "TRACKING PRESET",
        ["ISS vs Cosmos-2251 (Hypervelocity Crossing)", "CSS (Tianhe) vs Fengyun Fragment"]
    )

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    sensor_noise = st.slider("RADAR / OPTICAL NOISE (σ km)", 0.05, 1.5, 0.25, step=0.05)
    remaining_fuel = st.slider("AVAILABLE ΔV THRUST UNITS", 0.0, 100.0, 85.0, step=5.0)
    enable_edge_triage = st.checkbox("ACTIVATE ONBOARD FPGA TRIAGE", value=True)

# ---------------------------------------------------------------------------
# Orbital Math & Conjunction Engine
# ---------------------------------------------------------------------------
now_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

# SGP4 Real Satellites
l1_iss = "1 25544U 98067A   24080.52847222  .00016717  00000-0  10270-3 0  9014"
l2_iss = "2 25544  51.6416 179.3783 0004948 290.4121 161.2721 15.49842410444344"
iss_tle = parse_tle_lines("ISS (ZARYA)", l1_iss, l2_iss)
satellite = SpaceObject.from_tle(iss_tle, epoch=now_utc)

# Controlled close-proximity collision scenario for dynamic visualization
miss_distance_km = 0.38
rel_velocity_km_s = 14.28

# Bayesian Risk Calculation
p_coll = calculate_collision_probability(
    miss_distance_km=miss_distance_km,
    sigma_km=sensor_noise,
    hard_body_radius_m=6.5
)

# Autonomous State Machine
decision = evaluate_risk(
    p_collision=p_coll,
    miss_distance_km=miss_distance_km,
    fuel_remaining_delta_v=remaining_fuel
)

# A* Evasive Path Planner
planner = AStarPathPlanner(step_size=0.4, safety_margin_km=1.2)
sat_coords = np.array([0.0, 0.0, 0.0])
deb_coords = np.array([2.5, 0.25, 0.0])
target_coords = np.array([5.5, 0.0, 0.0])

evasive_path = planner.plan_evasive_path(sat_coords, target_coords, [deb_coords])
delta_v_cost = planner.calculate_path_delta_v(evasive_path)

# ---------------------------------------------------------------------------
# Hero Layout: Left Editorial Typography + Right 3D Planet & Orbit Visual
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.1, 1.9], gap="large")

with left_col:
    st.markdown('<div class="hero-tag">ASTRONOMICAL RECONNAISSANCE // CONJUNCTION 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Orbital<br>Deflection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-desc">'
        'Autonomous orbital safety platform executing real-time SGP4 ephemeris propagation, '
        'probabilistic Bayesian risk evaluation, and onboard edge ML telemetry triage to safeguard '
        'active spacecraft against hypervelocity kinetic collisions.'
        '</div>',
        unsafe_allow_html=True
    )

    # Status Pill
    if decision in ["EMERGENCY_AVOIDANCE", "CRITICAL_NO_FUEL_ALERT"]:
        pill_html = f'<div class="alert-emergency">CRITICAL HAZARD: {decision}</div>'
    elif decision == "PLAN_EVASIVE_MANEUVER":
        pill_html = f'<div class="alert-warning">DEFLECTION REQUIRED: {decision}</div>'
    else:
        pill_html = f'<div class="alert-safe">STATUS NOMINAL: {decision}</div>'
    st.markdown(pill_html, unsafe_allow_html=True)
    st.write("")

    # HUD Micro-Metrics
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">COLLISION PROB (P_C)</div>
            <div class="hud-value" style="color: #f43f5e;">{p_coll:.2e}</div>
            <div class="hud-sub">Hard sphere R = 6.5m</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">MISS DISTANCE (TCA)</div>
            <div class="hud-value" style="color: #38bdf8;">{miss_distance_km:.2f} <span style="font-size: 16px;">KM</span></div>
            <div class="hud-sub">Rel Vel: {rel_velocity_km_s:.1f} km/s</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">EVASIVE BURN (ΔV)</div>
            <div class="hud-value" style="color: #10b981;">{delta_v_cost:.1f} <span style="font-size: 16px;">M/S</span></div>
            <div class="hud-sub">A* 3D Optimal Detour</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">REMAINING FUEL</div>
            <div class="hud-value" style="color: #fbbf24;">{remaining_fuel:.0f} <span style="font-size: 16px;">ΔV</span></div>
            <div class="hud-sub">Hydrazine RCS Reserves</div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    # -----------------------------------------------------------------------
    # Cinematic 3D Celestial Globe & Conjunction Trajectory
    # -----------------------------------------------------------------------
    fig = go.Figure()

    # 1. 3D Planetary Sphere with Atmospheric Rim Shading
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    r_planet = 2.0  # Planetary scale

    # Position the planet off-center like the reference UI posters
    p_center_x, p_center_y, p_center_z = 2.8, -1.0, -1.2
    x_sphere = p_center_x + r_planet * np.outer(np.cos(theta), np.sin(phi))
    y_sphere = p_center_y + r_planet * np.outer(np.sin(theta), np.sin(phi))
    z_sphere = p_center_z + r_planet * np.outer(np.ones(np.size(theta)), np.cos(phi))

    # Planetary surface with Mars/Earth deep sci-fi gradient
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[
            [0.0, '#0c1b33'],
            [0.2, '#1e3a8a'],
            [0.4, '#0284c7'],
            [0.7, '#f97316'],
            [0.9, '#ea580c'],
            [1.0, '#ffffff']
        ],
        showscale=False,
        opacity=0.95,
        hoverinfo='skip',
        name='Earth / Mars Planet Body'
    ))

    # 2. Orbital Ring Trajectory
    orbit_t = np.linspace(0, 2 * np.pi, 80)
    orbit_x = p_center_x + (r_planet + 1.2) * np.cos(orbit_t)
    orbit_y = p_center_y + (r_planet + 1.2) * np.sin(orbit_t)
    orbit_z = p_center_z + 0.6 * np.cos(orbit_t)

    fig.add_trace(go.Scatter3d(
        x=orbit_x, y=orbit_y, z=orbit_z,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.15)', width=2, dash='dot'),
        hoverinfo='skip',
        name='Standard Keplerian Orbit'
    ))

    # 3. Spacecraft Marker (ISS)
    fig.add_trace(go.Scatter3d(
        x=[sat_coords[0]], y=[sat_coords[1]], z=[sat_coords[2]],
        mode='markers+text',
        marker=dict(size=10, color='#38bdf8', symbol='circle', line=dict(color='#ffffff', width=2)),
        text=['PRIMARY SPACECRAFT (ISS)'],
        textfont=dict(family='JetBrains Mono', size=11, color='#38bdf8'),
        textposition='top center',
        name='ISS (Zarya)'
    ))

    # 4. Debris Fragment Hazard Marker
    fig.add_trace(go.Scatter3d(
        x=[deb_coords[0]], y=[deb_coords[1]], z=[deb_coords[2]],
        mode='markers+text',
        marker=dict(size=8, color='#ef4444', symbol='diamond', line=dict(color='#fca5a5', width=1.5)),
        text=['DEBRIS // COSMOS-2251'],
        textfont=dict(family='JetBrains Mono', size=11, color='#ef4444'),
        textposition='bottom center',
        name='Cosmos-2251 Debris'
    ))

    # 5. Red Hazard Keep-out Bubble around debris
    u_h = np.linspace(0, 2 * np.pi, 16)
    v_h = np.linspace(0, np.pi, 16)
    r_hazard = 0.85
    hx = deb_coords[0] + r_hazard * np.outer(np.cos(u_h), np.sin(v_h))
    hy = deb_coords[1] + r_hazard * np.outer(np.sin(u_h), np.sin(v_h))
    hz = deb_coords[2] + r_hazard * np.outer(np.ones(np.size(u_h)), np.cos(v_h))

    fig.add_trace(go.Surface(
        x=hx, y=hy, z=hz,
        colorscale=[[0, 'rgba(239, 68, 68, 0.1)'], [1, 'rgba(239, 68, 68, 0.25)']],
        showscale=False,
        name='Keep-Out Safety Sphere'
    ))

    # 6. A* Fuel-Optimal Evasive Detour (Glowing Green Arc)
    path_np = np.array(evasive_path)
    fig.add_trace(go.Scatter3d(
        x=path_np[:, 0], y=path_np[:, 1], z=path_np[:, 2],
        mode='lines+markers',
        line=dict(color='#10b981', width=6),
        marker=dict(size=4, color='#34d399'),
        name='A* Evasive Path'
    ))

    # Dark sci-fi camera configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.3, y=-1.5, z=1.1),
                center=dict(x=0.8, y=0.0, z=0.0)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=580,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Bottom Section: Onboard Edge AI Data Triage (FPGA Telemetry Feed)
# ---------------------------------------------------------------------------
if enable_edge_triage:
    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="hero-tag">// REAL-TIME EDGE COMPUTING LAYER</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family: Cinzel; font-size: 26px; color: #f8fafc; letter-spacing: 2px; margin-bottom: 8px;">Onboard Neural Data Triage</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; font-size: 14px; margin-bottom: 20px;">Lightweight onboard edge classifier filtering obscured/blank optical frames directly in space prior to downlink.</p>', unsafe_allow_html=True)

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

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">FRAMES EVALUATED</div>
            <div class="hud-value">{rep['total_frames_processed']}</div>
            <div class="hud-sub">Edge FPGA Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">RAW OPTICAL CAPTURE</div>
            <div class="hud-value">{rep['total_raw_data_mb']} <span style="font-size: 16px;">MB</span></div>
            <div class="hud-sub">Unfiltered Sensor Memory</div>
        </div>
        """, unsafe_allow_html=True)
    with tc3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">GROUND DOWNLINK</div>
            <div class="hud-value" style="color: #38bdf8;">{rep['actual_downlinked_data_mb']} <span style="font-size: 16px;">MB</span></div>
            <div class="hud-sub">Priority Telemetry Queue</div>
        </div>
        """, unsafe_allow_html=True)
    with tc4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">BANDWIDTH SAVED</div>
            <div class="hud-value" style="color: #10b981;">{rep['bandwidth_saved_pct']}</div>
            <div class="hud-sub">{rep['bandwidth_saved_mb']} MB Filtered Onboard</div>
        </div>
        """, unsafe_allow_html=True)
