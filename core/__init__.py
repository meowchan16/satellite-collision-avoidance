"""
Core Engine Package for Satellite Collision Risk & Debris Avoidance Simulator.
Contains physics, Bayesian risk estimation, A* path planning, and decision logic.
"""

from core.physics import SpaceObject, PhysicsEngine
from core.data_ingestion import (
    TLEObject,
    CatalogManager,
    validate_tle_checksum,
    parse_tle_catalog,
    parse_tle_lines,
    classify_orbit,
)
from core.conjunction import find_tca
from core.bayesian_risk import calculate_collision_probability, BayesianRiskEngine
from core.decision_engine import evaluate_risk, DecisionEngine, DecisionState
from core.edge_triage import OnboardSensorFrame, EdgeAITriageEngine, TriageDecision
from core.maneuver_sim import execute_evasive_maneuver, calculate_new_trajectory

__all__ = [
    "SpaceObject",
    "PhysicsEngine",
    "TLEObject",
    "CatalogManager",
    "validate_tle_checksum",
    "parse_tle_catalog",
    "parse_tle_lines",
    "classify_orbit",
    "find_tca",
    "calculate_collision_probability",
    "BayesianRiskEngine",
    "evaluate_risk",
    "DecisionEngine",
    "DecisionState",
    "OnboardSensorFrame",
    "EdgeAITriageEngine",
    "TriageDecision",
    "execute_evasive_maneuver",
    "calculate_new_trajectory",
]

