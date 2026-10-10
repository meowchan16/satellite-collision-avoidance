"""
Comprehensive Automated Unit Tests for Teammate 2:
- Conjunction Assessment & Time of Closest Approach (TCA) (core/conjunction.py)
- Bayesian Collision Risk Engine (core/bayesian_risk.py)
- Autonomous Decision State Machine & Fuel Overrides (core/decision_engine.py)
"""

import math
from datetime import datetime, timezone
import pytest

from core.data_ingestion import parse_tle_lines
from core.physics import SpaceObject
from core.conjunction import find_tca
from core.bayesian_risk import calculate_collision_probability, BayesianRiskEngine
from core.decision_engine import evaluate_risk, DecisionEngine, DecisionState


# ---------------------------------------------------------------------------
# Test Fixtures & Sample TLEs
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_space_objects():
    """Provides two realistic SGP4-propagated space objects."""
    # ISS (ZARYA)
    l1_iss = "1 25544U 98067A   24080.52847222  .00016717  00000-0  10270-3 0  9014"
    l2_iss = "2 25544  51.6416 179.3783 0004948 290.4121 161.2721 15.49842410444344"
    iss_tle = parse_tle_lines("ISS (ZARYA)", l1_iss, l2_iss)
    iss = SpaceObject.from_tle(iss_tle)

    # Cosmos 2251 Debris
    l1_deb = "1 34123U 93036AR  24079.91234567  .00000452  00000-0  85231-4 0  9997"
    l2_deb = "2 34123  74.0321  45.1234 0021456 120.3456 240.5678 14.32145678789018"
    deb_tle = parse_tle_lines("COSMOS 2251 DEB", l1_deb, l2_deb)
    deb = SpaceObject.from_tle(deb_tle)

    return iss, deb


# ---------------------------------------------------------------------------
# 1. Tests for calculate_collision_probability
# ---------------------------------------------------------------------------
def test_collision_probability_far_away():
    """Far distance (500 km) with 0.5 km sigma should yield probability ~ 0."""
    p = calculate_collision_probability(miss_distance_km=500.0, sigma_km=0.5, hard_body_radius_m=3.0)
    assert p == 0.0 or p < 1e-30


def test_collision_probability_close_encounter():
    """
    Close encounter (0.2 km miss distance with 0.1 km sigma and 10m hard body radius)
    should produce a significant probability spike.
    """
    p_close = calculate_collision_probability(
        miss_distance_km=0.2,
        sigma_km=0.1,
        hard_body_radius_m=10.0
    )
    # R = 0.01 km, sigma = 0.1 km -> scale = 0.0001 / 0.02 = 0.005
    # exp(-0.04 / 0.02) = exp(-2) ~ 0.1353 -> p ~ 0.005 * 0.1353 ~ 6.7e-4
    assert p_close > 1e-4

    # Direct hit (0.0 km miss distance) must be strictly greater than a 0.2 km pass
    p_direct = calculate_collision_probability(
        miss_distance_km=0.0,
        sigma_km=0.1,
        hard_body_radius_m=10.0
    )
    assert p_direct > p_close
    assert 0.0 < p_direct <= 1.0


def test_collision_probability_edge_cases():
    """Guards against invalid inputs (negative/zero sigma, negative distance)."""
    with pytest.raises(ValueError, match="sigma_km must be strictly positive"):
        calculate_collision_probability(miss_distance_km=1.0, sigma_km=0.0)

    with pytest.raises(ValueError, match="sigma_km must be strictly positive"):
        calculate_collision_probability(miss_distance_km=1.0, sigma_km=-0.5)

    with pytest.raises(ValueError, match="miss_distance_km must be non-negative"):
        calculate_collision_probability(miss_distance_km=-2.0, sigma_km=0.5)

    # 0 hard-body radius should return 0.0 probability
    assert calculate_collision_probability(miss_distance_km=0.1, sigma_km=0.5, hard_body_radius_m=0.0) == 0.0


def test_bayesian_risk_engine_wrapper():
    """Verifies BayesianRiskEngine class wrapper backward compatibility."""
    engine = BayesianRiskEngine(sensor_sigma_pos=0.2)
    # Using scalar call
    p_scalar = engine.calculate_collision_probability(sat_pos=0.1, debris_pos=0.2, hard_sphere_radius=5.0)
    assert p_scalar > 0.0

    # Using 3D vectors
    p_vec = engine.calculate_collision_probability(
        sat_pos=[0.0, 0.0, 0.0],
        debris_pos=[0.1, 0.0, 0.0],
        hard_sphere_radius=5.0
    )
    assert p_vec > 0.0


# ---------------------------------------------------------------------------
# 2. Tests for evaluate_risk: 4 Primary Threshold States
# ---------------------------------------------------------------------------
def test_evaluate_risk_maintain_course():
    """Negligible collision probability and safe distance -> MAINTAIN_COURSE."""
    state = evaluate_risk(
        p_collision=1e-6,
        miss_distance_km=25.0,
        fuel_remaining_delta_v=80.0
    )
    assert state == "MAINTAIN_COURSE"


def test_evaluate_risk_monitor_close_approach():
    """Probability between [1e-5, 1e-4) -> MONITOR_CLOSE_APPROACH."""
    state = evaluate_risk(
        p_collision=5e-5,
        miss_distance_km=10.0,
        fuel_remaining_delta_v=80.0
    )
    assert state == "MONITOR_CLOSE_APPROACH"


def test_evaluate_risk_plan_evasive_maneuver():
    """Probability between [1e-4, 1e-2) -> PLAN_EVASIVE_MANEUVER."""
    state = evaluate_risk(
        p_collision=2e-3,
        miss_distance_km=5.0,
        fuel_remaining_delta_v=80.0
    )
    assert state == "PLAN_EVASIVE_MANEUVER"


def test_evaluate_risk_emergency_avoidance_by_probability():
    """Probability >= 1e-2 -> EMERGENCY_AVOIDANCE."""
    state = evaluate_risk(
        p_collision=0.05,
        miss_distance_km=5.0,
        fuel_remaining_delta_v=80.0
    )
    assert state == "EMERGENCY_AVOIDANCE"


def test_evaluate_risk_emergency_avoidance_by_distance():
    """Miss distance < 1.0 km -> EMERGENCY_AVOIDANCE regardless of p_collision."""
    state = evaluate_risk(
        p_collision=1e-7,
        miss_distance_km=0.6,
        fuel_remaining_delta_v=80.0
    )
    assert state == "EMERGENCY_AVOIDANCE"


# ---------------------------------------------------------------------------
# 3. Tests for evaluate_risk: CRITICAL_NO_FUEL_ALERT Override
# ---------------------------------------------------------------------------
def test_evaluate_risk_critical_no_fuel_override_on_emergency():
    """When state would be EMERGENCY_AVOIDANCE but fuel is 0 -> CRITICAL_NO_FUEL_ALERT."""
    state = evaluate_risk(
        p_collision=0.05,
        miss_distance_km=0.5,
        fuel_remaining_delta_v=0.0
    )
    assert state == "CRITICAL_NO_FUEL_ALERT"


def test_evaluate_risk_critical_no_fuel_override_on_plan_evasive():
    """When state would be PLAN_EVASIVE_MANEUVER but fuel is 0 -> CRITICAL_NO_FUEL_ALERT."""
    state = evaluate_risk(
        p_collision=5e-4,
        miss_distance_km=5.0,
        fuel_remaining_delta_v=0.0
    )
    assert state == "CRITICAL_NO_FUEL_ALERT"


def test_evaluate_risk_no_fuel_but_safe():
    """When fuel is 0 but path is completely safe, state remains MAINTAIN_COURSE."""
    state = evaluate_risk(
        p_collision=1e-8,
        miss_distance_km=100.0,
        fuel_remaining_delta_v=0.0
    )
    assert state == "MAINTAIN_COURSE"


def test_decision_engine_class_wrapper():
    """Verifies DecisionEngine class and DecisionState enum wrapper."""
    engine = DecisionEngine()
    result = engine.evaluate_risk(p_collision=0.05, remaining_fuel=50.0, miss_distance_km=2.0)
    assert result == DecisionState.EMERGENCY_AVOIDANCE
    assert result.value == "EMERGENCY_AVOIDANCE"


# ---------------------------------------------------------------------------
# 4. Tests for find_tca (Conjunction Assessment)
# ---------------------------------------------------------------------------
def test_find_tca_dictionary_structure(sample_space_objects):
    """Ensures find_tca propagates and returns expected dictionary keys and types."""
    sat, deb = sample_space_objects
    start_time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Short 15-minute test screening window
    result = find_tca(sat, deb, start_time, window_minutes=15.0)

    # Verify keys
    assert isinstance(result, dict)
    assert "tca_time" in result
    assert "miss_distance_km" in result
    assert "relative_velocity_km_s" in result

    # Verify data types and values
    assert isinstance(result["tca_time"], datetime)
    assert result["tca_time"].tzinfo == timezone.utc
    assert isinstance(result["miss_distance_km"], float)
    assert result["miss_distance_km"] > 0.0

    assert isinstance(result["relative_velocity_km_s"], float)
    assert result["relative_velocity_km_s"] > 0.0
    # Orbital relative velocity in LEO can reach up to ~15 km/s
    assert result["relative_velocity_km_s"] < 20.0
