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

__all__ = [
    "SpaceObject",
    "PhysicsEngine",
    "TLEObject",
    "CatalogManager",
    "validate_tle_checksum",
    "parse_tle_catalog",
    "parse_tle_lines",
    "classify_orbit",
]

