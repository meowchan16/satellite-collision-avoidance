import streamlit as st
import numpy as np
import plotly.graph_objects as go
from core.physics import SpaceObject, PhysicsEngine
from core.bayesian_risk import BayesianRiskEngine
from core.path_planner import AStarPathPlanner
from core.decision_engine import DecisionEngine, DecisionState

st.set_page_config(page_title="Satellite Collision Risk Simulator", layout="wide")

st.title("🛸 Autonomous Satellite Collision-Avoidance & Risk Simulator")
st.markdown("Integrates **Kinematic Physics**, **Bayesian Risk Estimation**, **A* Path Planning**, and **Rule-Based Decision Logic**.")

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Settings")
sensor_noise = st.sidebar.slider("Sensor Noise (σ km)", 0.1, 2.0, 0.5, step=0.1)
remaining_fuel = st.sidebar.slider("Remaining Fuel (Δv units)", 0.0, 100.0, 80.0, step=5.0)

# Initialize Core Engines
physics = PhysicsEngine(dt=1.0)
risk_engine = BayesianRiskEngine(sensor_sigma_pos=sensor_noise)
decision_engine = DecisionEngine()
planner = AStarPathPlanner()

# Instantiate Objects
satellite = SpaceObject("Sat-1", position=[0, 0, 0], velocity=[1, 0, 0], is_satellite=True)
debris = SpaceObject("Debris-A", position=[5, 0.5, 0], velocity=[-0.8, 0, 0])

# Perform 1 step risk assessment
p_coll = risk_engine.calculate_collision_probability(satellite.position, debris.position)
decision = decision_engine.evaluate_risk(p_coll, remaining_fuel)

# Layout Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Collision Probability (P_coll)", f"{p_coll:.6f}")
col2.metric("Autonomous Action State", decision.value)
col3.metric("Fuel Level", f"{remaining_fuel} Δv")

# Plotly 3D Visualization
fig = go.Figure()

# Satellite
fig.add_trace(go.Scatter3d(
    x=[satellite.position[0]], y=[satellite.position[1]], z=[satellite.position[2]],
    mode='markers+text', marker=dict(size=10, color='blue'), name='Satellite (Sat-1)',
    text=['Sat-1']
))

# Debris
fig.add_trace(go.Scatter3d(
    x=[debris.position[0]], y=[debris.position[1]], z=[debris.position[2]],
    mode='markers+text', marker=dict(size=8, color='red'), name='Debris (Debris-A)',
    text=['Debris-A']
))

fig.update_layout(
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=500
)

st.plotly_chart(fig, use_container_width=True)
