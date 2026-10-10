"""
Unit Tests for A* Path Planner.
Verifies collision avoidance, keep-out boundary compliance, and fuel estimation.
"""

import numpy as np
import pytest
from core.path_planner import AStarPathPlanner


def test_direct_path_when_no_obstacle():
    """Direct straight line should be chosen when no obstacles exist."""
    planner = AStarPathPlanner(step_size=0.5, safety_margin_km=1.0)
    start = np.array([0.0, 0.0, 0.0])
    target = np.array([5.0, 0.0, 0.0])
    debris = []

    path = planner.plan_evasive_path(start, target, debris)
    assert len(path) == 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[1], target)


def test_evasive_detour_around_debris():
    """Path must steer clear of debris center by at least safety_margin_km."""
    safety_margin = 1.0
    planner = AStarPathPlanner(step_size=0.5, safety_margin_km=safety_margin)
    start = np.array([0.0, 0.0, 0.0])
    target = np.array([6.0, 0.0, 0.0])
    debris_center = np.array([3.0, 0.0, 0.0])

    path = planner.plan_evasive_path(start, target, [debris_center])
    assert len(path) > 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], target)

    # Verify no waypoint enters the keep-out sphere
    for pt in path[1:-1]:
        dist = np.linalg.norm(pt - debris_center)
        assert dist >= safety_margin * 0.95, f"Waypoint {pt} too close to debris: {dist} km"


def test_delta_v_calculation():
    """Ensures calculate_path_delta_v returns non-negative fuel estimation."""
    planner = AStarPathPlanner()
    straight_path = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
    assert planner.calculate_path_delta_v(straight_path) == 0.0

    bent_path = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 2.0, 0.0]),
        np.array([10.0, 0.0, 0.0])
    ]
    dv = planner.calculate_path_delta_v(bent_path)
    assert dv > 0.0
