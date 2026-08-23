# 🛰️ Satellite Collision Risk & Debris Avoidance Simulator

An autonomous satellite collision-avoidance system integrating physics simulation, Bayesian risk estimation, A* path planning, and rule-based decision logic into a single real-time decision pipeline.

---

## 👥 Team & Component Breakdown (3 Collaborators)

To allow seamless parallel development without Git merge conflicts, the system is split into independent modular components:

| Collaborator | GitHub Profile | Assigned Component | Key Responsibilities & Files |
| :--- | :--- | :--- | :--- |
| **meowchan16** | [@meowchan16](https://github.com/meowchan16) | **A* Path Planner & UI Dashboard** | Fuel-optimal dynamic trajectory generation, interactive Streamlit & Plotly 3D visualizer (`core/path_planner.py`, `app.py`). |
| **Simran Bishnoi** | [@Simranbishnoi](https://github.com/Simranbishnoi) | **Physics & Kinematics Engine** | Orbital relative motion propagation, Clohessy-Wiltshire dynamics, trajectory tracking (`core/physics.py`). |
| **Greeshma** | [@greeshmaa965-cell](https://github.com/greeshmaa965-cell) | **Bayesian Risk & Decision Engine** | Sensor measurement noise propagation, $P_{\text{collision}}$ calculation, rule-based state machine (`core/bayesian_risk.py`, `core/decision_engine.py`). |

---

## 🛠️ System Architecture

1. **Physics Engine (`core/physics.py`)**: Models 3D relative positions and velocities over time.
2. **Bayesian Risk Engine (`core/bayesian_risk.py`)**: Models sensor measurement noise ($\sigma$) into continuous collision probability ($P_{\text{collision}}$).
3. **Decision Engine (`core/decision_engine.py`)**: Rule-based state machine evaluating risk thresholds (`MAINTAIN_COURSE`, `PLAN_EVASIVE_MANEUVER`, `EXECUTE_EVASIVE_MANEUVER`, `EMERGENCY_ABORT`).
4. **Path Planner (`core/path_planner.py`)**: Executes A* search to calculate fuel-efficient evasive maneuvers around debris.
5. **Interactive Dashboard (`app.py`)**: Renders real-time telemetry, 3D Plotly trajectory views, risk gauges, and simulation controls.

---

## 🌿 Git Setup & Team Workflow

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/meowchan16/satellite-collision-avoidance.git
cd satellite-collision-avoidance

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Team Branching Strategy

Each team member works on a dedicated feature branch:

```bash
# Create and switch to your component branch
git checkout -b feature/physics-engine      # Member 1
git checkout -b feature/bayesian-risk       # Member 2
git checkout -b feature/path-planner-ui     # Member 3
```

### 3. Pushing Changes & Pull Requests

```bash
# Commit your work
git add .
git commit -m "Add feature description"

# Push feature branch to GitHub
git push origin feature/<your-branch-name>
```
*Open a Pull Request (PR) on GitHub to review and merge code into `main`.*

---

## 🚀 Running the Simulator Dashboard

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
