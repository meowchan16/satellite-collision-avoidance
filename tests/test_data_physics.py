"""
Automated unit tests for Teammate 1 responsibilities:
- Data Ingestion & TLE Checksum Validation (core/data_ingestion.py)
- SGP4 Orbit State Propagation (core/physics.py)
- Unified 3D State Vectors r=[x, y, z] km, v=[vx, vy, vz] km/s
- Sensor Noise Layer (Gaussian noise injection)
"""

import math
from datetime import datetime, timezone
import numpy as np
import pytest

from core.data_ingestion import (
    validate_tle_checksum,
    parse_tle_lines,
    parse_tle_catalog,
    CatalogManager,
    classify_orbit,
)
from core.physics import SpaceObject, PhysicsEngine


# Real verified ISS TLE with exact modulo-10 checksum
ISS_LINE1 = "1 25544U 98067A   24080.52847222  .00016717  00000-0  10270-3 0  9014"
ISS_LINE2 = "2 25544  51.6416 179.3783 0004948 290.4121 161.2721 15.49842410444344"
ISS_NAME = "ISS (ZARYA)"

SAMPLE_CATALOG_RAW = f"""{ISS_NAME}
{ISS_LINE1}
{ISS_LINE2}
COSMOS 2251 DEB
1 34123U 93036AR  24079.91234567  .00000452  00000-0  85231-4 0  9997
2 34123  74.0321  45.1234 0021456 120.3456 240.5678 14.32145678789018
"""


def test_validate_tle_checksum_valid():
    assert validate_tle_checksum(ISS_LINE1) is True
    assert validate_tle_checksum(ISS_LINE2) is True


def test_validate_tle_checksum_corrupted():
    # Corrupt a character or the checksum digit
    corrupted_line1 = ISS_LINE1[:-1] + ("0" if ISS_LINE1[-1] != "0" else "1")
    assert validate_tle_checksum(corrupted_line1) is False

    # Short line
    assert validate_tle_checksum("1 25544U short") is False


def test_classify_orbit():
    # LEO: mean motion ~ 15 revs/day
    assert classify_orbit(15.5, 0.001) == "LEO"
    # GEO: mean motion ~ 1.0 rev/day
    assert classify_orbit(1.002, 0.0005) == "GEO"
    # MEO: GPS ~ 2.0 rev/day
    assert classify_orbit(2.0, 0.01) == "MEO"
    # HEO: high eccentricity
    assert classify_orbit(2.0, 0.7) == "HEO"


def test_parse_tle_lines():
    tle_obj = parse_tle_lines(ISS_NAME, ISS_LINE1, ISS_LINE2)
    assert tle_obj.name == ISS_NAME
    assert tle_obj.norad_id == 25544
    assert tle_obj.object_type == "PAYLOAD"
    assert tle_obj.orbit_class == "LEO"
    assert math.isclose(tle_obj.inclination_deg, 51.6416, rel_tol=1e-4)
    assert math.isclose(tle_obj.mean_motion_rev_day, 15.4984241, rel_tol=1e-4)
    assert tle_obj.is_valid_checksum() is True


def test_catalog_manager_ingestion_and_filtering():
    manager = CatalogManager()
    count = manager.ingest_text(SAMPLE_CATALOG_RAW, validate_checksums=True)
    assert count == 2
    assert len(manager) == 2

    # Query by NORAD ID
    iss = manager.get_by_norad_id(25544)
    assert iss is not None
    assert iss.name == ISS_NAME

    # Filter by type
    payloads = manager.filter_by_type("PAYLOAD")
    assert len(payloads) == 1
    assert payloads[0].norad_id == 25544

    debris = manager.filter_by_type("DEBRIS")
    assert len(debris) == 1
    assert debris[0].norad_id == 34123

    # Filter by orbit class
    leo_objs = manager.filter_by_orbit_class("LEO")
    assert len(leo_objs) == 2


def test_sgp4_orbit_propagation_physics():
    """
    Validates SGP4 produces valid 3D Cartesian state vectors r=[x, y, z] km, v=[vx, vy, vz] km/s
    consistent with Low Earth Orbit physics.
    """
    tle_obj = parse_tle_lines(ISS_NAME, ISS_LINE1, ISS_LINE2)
    sat = SpaceObject.from_tle(tle_obj, epoch=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc))

    r, v = sat.position, sat.velocity

    # Check state vector dimensions and types
    assert isinstance(r, np.ndarray) and r.shape == (3,)
    assert isinstance(v, np.ndarray) and v.shape == (3,)

    # Low Earth Orbit radius check: Earth radius ~6378 km + ISS altitude ~415 km -> distance ~6793 km
    r_mag = np.linalg.norm(r)
    assert 6500.0 <= r_mag <= 7000.0, f"Orbital radius {r_mag} km is unphysical for ISS LEO"

    # LEO orbital speed check: v ~ sqrt(GM/r) ~ 7.66 km/s
    v_mag = np.linalg.norm(v)
    assert 7.0 <= v_mag <= 8.2, f"Orbital velocity {v_mag} km/s is unphysical for ISS LEO"

    # Propagate 60 seconds forward
    target_time = datetime(2024, 3, 20, 12, 1, 0, tzinfo=timezone.utc)
    r2, v2 = sat.propagate_sgp4(target_time)

    # Position must have moved by approximately ~ 60s * 7.66 km/s ~ 460 km
    delta_pos = np.linalg.norm(r2 - r)
    assert 400.0 <= delta_pos <= 500.0, f"Expected ~460 km displacement in 60s, got {delta_pos:.2f} km"


def test_sensor_noise_layer():
    """
    Validates that the sensor noise layer models real-world tracking uncertainty
    by injecting configurable Gaussian noise N(0, sigma^2).
    """
    tle_obj = parse_tle_lines(ISS_NAME, ISS_LINE1, ISS_LINE2)
    sat = SpaceObject.from_tle(tle_obj, epoch=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc))

    sigma_pos = 0.5  # 500 meters
    sigma_vel = 0.002  # 2 m/s

    rng = np.random.default_rng(42)
    noisy_pos, noisy_vel = sat.get_noisy_observation(sigma_pos=sigma_pos, sigma_vel=sigma_vel, rng=rng)

    # Check dimensions
    assert noisy_pos.shape == (3,)
    assert noisy_vel.shape == (3,)

    # Statistical verification across N observations
    n_samples = 2000
    pos_errors = []
    vel_errors = []
    for _ in range(n_samples):
        obs_p, obs_v = sat.get_noisy_observation(sigma_pos=sigma_pos, sigma_vel=sigma_vel, rng=rng)
        pos_errors.append(obs_p - sat.position)
        vel_errors.append(obs_v - sat.velocity)

    pos_errors = np.array(pos_errors)
    vel_errors = np.array(vel_errors)

    # Check mean error is close to 0
    assert np.all(np.abs(np.mean(pos_errors, axis=0)) < 0.05)
    # Check standard deviation matches sigma
    emp_sigma_pos = np.std(pos_errors, axis=0)
    for s in emp_sigma_pos:
        assert math.isclose(s, sigma_pos, rel_tol=0.15)


def test_physics_engine_batch_propagation():
    """Validates PhysicsEngine can batch propagate multiple objects with SGP4."""
    manager = CatalogManager()
    manager.ingest_text(SAMPLE_CATALOG_RAW)
    objects = [SpaceObject.from_tle(item, epoch=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)) for item in manager.get_all()]

    engine = PhysicsEngine()
    future_time = datetime(2024, 3, 20, 12, 10, 0, tzinfo=timezone.utc)
    results = engine.propagate_all_sgp4(objects, future_time)

    assert len(results) == 2
    for r, v in results:
        assert len(r) == 3
        assert len(v) == 3
        assert np.linalg.norm(r) > 6000.0


def test_backward_compatibility_space_object():
    """Validates that existing code instantiating SpaceObject without TLE continues to work."""
    obj = SpaceObject("Sat-1", position=[0, 0, 0], velocity=[1, 0, 0], is_satellite=True)
    assert obj.name == "Sat-1"
    assert obj.is_satellite is True
    assert np.allclose(obj.position, [0, 0, 0])

    # Test kinematic update
    obj.update_position(dt=2.0)
    assert np.allclose(obj.position, [2, 0, 0])
    assert len(obj.trajectory_history) == 2
