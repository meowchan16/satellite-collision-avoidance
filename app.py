import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

from core.physics import SpaceObject, PhysicsEngine
from core.data_ingestion import parse_tle_lines, CatalogManager
from core.conjunction import find_tca
from core.bayesian_risk import calculate_collision_probability, BayesianRiskEngine
from core.decision_engine import evaluate_risk, DecisionState
from core.path_planner import AStarPathPlanner
from core.edge_triage import OnboardSensorFrame, EdgeAITriageEngine

# Page configuration
st.set_page_config(
    page_title="Autonomous SSA & Space Debris Avoidance System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<div style="background: linear-gradient(135deg, #0b192c 0%, #1e3e62 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px; border: 1px solid #3b82f6;">
    <h1 style="color: #60a5fa; margin: 0; font-size: 28px;">🛰️ Autonomous Satellite Collision-Avoidance & Risk Simulator</h1>
    <p style="color: #cbd5e1; margin: 5px 0 0 0; font-size: 15px;">
        Integrating <b>SGP4 Orbital Astrodynamics</b>, <b>Conjunction Assessment (TCA)</b>, <b>Bayesian Risk Estimation</b>, <b>3D A* Path Planning</b>, and <b>Onboard Edge AI Data Triage</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Mission Control Parameters")

scenario = st.sidebar.selectbox(
    "Encounter Scenario",
    [
        "ISS vs COSMOS-2251 Fragment (Critical Conjunction)",
        "CSS (Tianhe) vs Fengyun-1C Debris (Close Approach)",
        "Custom Orbital Kinematics (Test Simulation)"
    ]
)

st.sidebar.subheader("Radar & Optical Sensor Noise")
sensor_noise = st.sidebar.slider("Tracking Uncertainty (σ km)", 0.05, 2.0, 0.2, step=0.05)
hard_body_radius = st.sidebar.slider("Hard-Body Collision Radius (m)", 1.0, 15.0, 5.0, step=0.5)

st.sidebar.subheader("Spacecraft Propulsion")
remaining_fuel = st.sidebar.slider("Remaining Fuel (Δv units)", 0.0, 100.0, 75.0, step=5.0)

st.sidebar.subheader("Onboard Edge AI Triage")
simulate_edge_triage = st.sidebar.checkbox("Simulate Real-time Edge AI Data Triage", value=True)

# ---------------------------------------------------------------------------
# Setup Space Objects & Orbital Conjunction
# ---------------------------------------------------------------------------
now_utc = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

if "ISS" in scenario:
    l1_sat = "1 25544U 98067A   24080.52847222  .00016717  00000-0  10270-3 0  9014"
    l2_sat = "2 25544  51.6416 179.3783 0004948 290.4121 161.2721 15.49842410444344"
    sat_name = "ISS (ZARYA)"

    # Shift debris closer for dramatic risk scenario
    l1_deb = "1 34123U 93036AR  24079.91234567  .00000452  00000-0  85231-4 0  9997"
    l2_deb = "2 34123  74.0321  45.1234 0021456 120.3456 240.5678 14.32145678789018"
    deb_name = "COSMOS-2251 DEB"

    sat_tle = parse_tle_lines(sat_name, l1_sat, l2_sat)
    deb_tle = parse_tle_lines(deb_name, l1_deb, l2_deb)
    satellite = SpaceObject.from_tle(sat_tle, epoch=now_utc)
    debris = SpaceObject.from_tle(deb_tle, epoch=now_utc)

    # Compute TCA
    conjunction_data = find_tca(satellite, debris, now_utc, window_minutes=15.0)
    tca_time = conjunction_data["tca_time"]
    miss_distance_km = 0.45  # Controlled scenario encounter for avoidance demonstration
    rel_vel = conjunction_data["relative_velocity_km_s"]
else:
    sat_name = "Sat-1 (Active)"
    deb_name = "Debris Fragment A"
    satellite = SpaceObject(sat_name, position=[0.0, 0.0, 0.0], velocity=[7.5, 0.0, 0.0], is_satellite=True)
    debris = SpaceObject(deb_name, position=[3.0, 0.3, 0.0], velocity=[-6.0, 0.0, 0.0])
    tca_time = now_utc
    miss_distance_km = 0.3
    rel_vel = 13.5

# ---------------------------------------------------------------------------
# Calculate Bayesian Collision Probability & Autonomous Decision
# ---------------------------------------------------------------------------
p_coll = calculate_collision_probability(
    miss_distance_km=miss_distance_km,
    sigma_km=sensor_noise,
    hard_body_radius_m=hard_body_radius
)

decision_state = evaluate_risk(
    p_collision=p_coll,
    miss_distance_km=miss_distance_km,
    fuel_remaining_delta_v=remaining_fuel
)

# ---------------------------------------------------------------------------
# Compute A* Evasive Maneuver if needed
# ---------------------------------------------------------------------------
planner = AStarPathPlanner(step_size=0.5, safety_margin_km=1.2)
sat_pos = satellite.position
deb_pos = debris.position

# Target position forward along trajectory
flight_vec = np.array([6.0, 0.0, 0.0])
target_pos = sat_pos + flight_vec

if decision_state in [DecisionState.PLAN_EVASIVE_MANEUVER.value, DecisionState.EMERGENCY_AVOIDANCE.value]:
    evasive_path = planner.plan_evasive_path(sat_pos, target_pos, [deb_pos])
    estimated_dv = planner.calculate_path_delta_v(evasive_path)
else:
    evasive_path = [sat_pos, target_pos]
    estimated_dv = 0.0

# ---------------------------------------------------------------------------
# Metric Row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Collision Probability (P_c)", f"{p_coll:.2e}")
col2.metric("Miss Distance at TCA", f"{miss_distance_km:.2f} km")

# Color badge for decision state
state_colors = {
    "MAINTAIN_COURSE": "🟢",
    "MONITOR_CLOSE_APPROACH": "🟡",
    "PLAN_EVASIVE_MANEUVER": "🟠",
    "EMERGENCY_AVOIDANCE": "🔴",
    "CRITICAL_NO_FUEL_ALERT": "🚨"
}
col3.metric("Autonomous Flight State", f"{state_colors.get(decision_state, '⚪')} {decision_state}")
col4.metric("Evasive Maneuver Δv", f"{estimated_dv:.1f} m/s", delta=f"{remaining_fuel} Δv left")

st.divider()

# ---------------------------------------------------------------------------
# 3D Interactive Encounter Visualization
# ---------------------------------------------------------------------------
st.subheader("🌐 3D Spatial Conjunction & Evasive Trajectory Visualization")

fig = go.Figure()

# Satellite marker
fig.add_trace(go.Scatter3d(
    x=[sat_pos[0]], y=[sat_pos[1]], z=[sat_pos[2]],
    mode='markers+text',
    marker=dict(size=9, color='#3b82f6', symbol='circle'),
    name=f'Primary Satellite ({sat_name})',
    text=[sat_name],
    textposition="top center"
))

# Debris marker
fig.add_trace(go.Scatter3d(
    x=[deb_pos[0]], y=[deb_pos[1]], z=[deb_pos[2]],
    mode='markers+text',
    marker=dict(size=8, color='#ef4444', symbol='diamond'),
    name=f'Tracked Debris ({deb_name})',
    text=[deb_name],
    textposition="top center"
))

# Debris Keep-out Sphere (Wireframe bubble)
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
sphere_r = 1.0  # km
xs = deb_pos[0] + sphere_r * np.outer(np.cos(u), np.sin(v))
ys = deb_pos[1] + sphere_r * np.outer(np.sin(u), np.sin(v))
zs = deb_pos[2] + sphere_r * np.outer(np.ones(np.size(u)), np.cos(v))

fig.add_trace(go.Surface(
    x=xs, y=ys, z=zs,
    colorscale=[[0, 'rgba(239, 68, 68, 0.15)'], [1, 'rgba(239, 68, 68, 0.3)']],
    showscale=False,
    name='Hazard Keep-Out Sphere'
))

# Planned Evasive Path (A* Detour)
path_arr = np.array(evasive_path)
fig.add_trace(go.Scatter3d(
    x=path_arr[:, 0], y=path_arr[:, 1], z=path_arr[:, 2],
    mode='lines+markers',
    line=dict(color='#10b981', width=5),
    marker=dict(size=4, color='#10b981'),
    name='A* Evasive Trajectory Detour'
))

fig.update_layout(
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        bgcolor='#0f172a',
        xaxis=dict(backgroundcolor='#0f172a', gridcolor='#334155'),
        yaxis=dict(backgroundcolor='#0f172a', gridcolor='#334155'),
        zaxis=dict(backgroundcolor='#0f172a', gridcolor='#334155')
    ),
    paper_bgcolor='#0f172a',
    font=dict(color='#f8fafc'),
    margin=dict(l=0, r=0, b=0, t=20),
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Onboard Edge AI Data Triage Section
# ---------------------------------------------------------------------------
if simulate_edge_triage:
    st.divider()
    st.subheader("⚡ Onboard Edge AI Data Triage (FPGA / Neural Edge Processor)")
    st.caption("Live edge filter discarding cloud-obscured/blank sensor frames onboard to conserve ground station downlink bandwidth.")

    triage_engine = EdgeAITriageEngine()

    # Stream of sample telemetry captures
    stream_frames = [
        OnboardSensorFrame("FRAME_101", cloud_cover_pct=82.0, signal_to_noise_ratio=15.0, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("FRAME_102", cloud_cover_pct=12.0, signal_to_noise_ratio=24.0, debris_streak_detected=True, raw_size_mb=45.0),
        OnboardSensorFrame("FRAME_103", cloud_cover_pct=91.0, signal_to_noise_ratio=11.0, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("FRAME_104", cloud_cover_pct=05.0, signal_to_noise_ratio=0.8, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("FRAME_105", cloud_cover_pct=22.0, signal_to_noise_ratio=19.0, debris_streak_detected=False, raw_size_mb=45.0),
        OnboardSensorFrame("FRAME_106", cloud_cover_pct=76.0, signal_to_noise_ratio=14.0, debris_streak_detected=False, raw_size_mb=45.0),
    ]

    for frame in stream_frames:
        triage_engine.classify_frame(frame)

    triage_report = triage_engine.get_savings_report()

    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Frames Evaluated Onboard", triage_report["total_frames_processed"])
    tcol2.metric("Raw Optical Data Captured", f"{triage_report['total_raw_data_mb']} MB")
    tcol3.metric("Downlinked to Ground Station", f"{triage_report['actual_downlinked_data_mb']} MB")
    tcol4.metric("Bandwidth Conserved", f"{triage_report['bandwidth_saved_mb']} MB", delta=f"{triage_report['bandwidth_saved_pct']} Saved")

    with st.expander("🔍 View Edge Inference Frame Classification Log"):
        st.json(triage_report["decision_counts"])
