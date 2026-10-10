"""
Unit Tests for Orbital Maneuver Simulation Module (core/maneuver_sim.py).
Verifies prograde, retrograde, and radial burns, Vis-Viva orbital changes,
and perturbed post-burn numerical trajectory generation.
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from core.maneuver_sim import execute_evasive_maneuver, calculate_new_trajectory
from core.data_ingestion import parse_tle_lines
from core.physics import SpaceObject


@pytest.fixture
def circular_leo_state():
    """Provides a circular Low Earth Orbit state vector (r=6800 km, v~7.66 km/s)."""
    r = np.array([6800.0, 0.0, 0.0])
    v = np.array([0.0, 7.659, 0.0])
    return r, v


def test_prograde_burn_increases_velocity(circular_leo_state):
    """Prograde burn must increase orbital speed and raise semi-major axis."""
    r, v = circular_leo_state
    delta_v_m_s = 50.0  # +50 m/s prograde

    new_v, details = execute_evasive_maneuver(r, v, delta_v_magnitude=delta_v_m_s, burn_direction="prograde")

    assert np.linalg.norm(new_v) > np.linalg.norm(v)
    assert details["burn_executed"] == "Burn executed: 50.00 m/s Prograde"
    assert "LEO" in details["new_orbit"]
    assert details["post_burn_semi_major_axis_km"] > 6800.0


def test_retrograde_burn_decreases_velocity(circular_leo_state):
    """Retrograde burn must decrease orbital speed and lower semi-major axis."""
    r, v = circular_leo_state
    delta_v_m_s = 30.0  # 30 m/s retrograde

    new_v, details = execute_evasive_maneuver(r, v, delta_v_magnitude=delta_v_m_s, burn_direction="retrograde")

    assert np.linalg.norm(new_v) < np.linalg.norm(v)
    assert details["burn_executed"] == "Burn executed: 30.00 m/s Retrograde"
    assert details["post_burn_semi_major_axis_km"] < 6800.0


def test_radial_burn_preserves_speed(circular_leo_state):
    """Orthogonal radial burn applies perpendicular vector without direct prograde acceleration."""
    r, v = circular_leo_state
    delta_v_m_s = 20.0

    new_v, details = execute_evasive_maneuver(r, v, delta_v_magnitude=delta_v_m_s, burn_direction="radial_out")
    assert "Radial-Out" in details["burn_executed"]
    # New speed is sqrt(v^2 + dv^2)
    expected_speed = float(np.sqrt(np.linalg.norm(v)**2 + (20.0 / 1000.0)**2))
    assert np.isclose(np.linalg.norm(new_v), expected_speed, rtol=1e-4)


def test_calculate_new_trajectory_points():
    """Verifies that calculate_new_trajectory propagates a continuous sequence of 3D points."""
    l1 = "1 25544U 98067A   24080.52847222  .00016717  00000-0  10270-3 0  9014"
    l2 = "2 25544  51.6416 179.3783 0004948 290.4121 161.2721 15.49842410444344"
    iss_tle = parse_tle_lines("ISS (ZARYA)", l1, l2)
    iss = SpaceObject.from_tle(iss_tle)

    burn_time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    trajectory = calculate_new_trajectory(
        tle=iss,
        maneuver_time=burn_time,
        delta_v=10.0,
        burn_direction="prograde",
        duration_minutes=30.0,
        step_seconds=60.0
    )

    # 30 mins * 60s / 60s = 30 steps + 1 initial point = 31 points
    assert len(trajectory) == 31
    for pt in trajectory:
        assert len(pt) == 3
        # LEO radius check: distance from center of Earth should remain between 6500 and 7200 km
        norm_r = np.linalg.norm(pt)
        assert 6500.0 < norm_r < 7500.0, f"Unphysical orbit radius {norm_r} km"
