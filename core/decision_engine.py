"""
Autonomous Decision Engine.
Evaluates continuous collision risk and proximity criteria against industry
safety thresholds and fuel constraints, outputting autonomous flight control states.
"""

from enum import Enum
from typing import Optional, Union


class DecisionState(str, Enum):
    """
    Autonomous Spacecraft Decision States.
    Inherits from str for seamless string comparisons and serialization.
    """
    MAINTAIN_COURSE = "MAINTAIN_COURSE"
    MONITOR_CLOSE_APPROACH = "MONITOR_CLOSE_APPROACH"
    PLAN_EVASIVE_MANEUVER = "PLAN_EVASIVE_MANEUVER"
    EMERGENCY_AVOIDANCE = "EMERGENCY_AVOIDANCE"
    CRITICAL_NO_FUEL_ALERT = "CRITICAL_NO_FUEL_ALERT"

    def __str__(self) -> str:
        return self.value


def evaluate_risk(
    p_collision: float,
    miss_distance_km: float,
    fuel_remaining_delta_v: float
) -> str:
    """
    Evaluates autonomous flight state based on collision probability, miss distance,
    and remaining fuel delta-v.

    Decision Rules:
      - If p_collision >= 1e-2 OR miss_distance_km < 1.0:
            Base state: "EMERGENCY_AVOIDANCE"
      - Elif p_collision >= 1e-4:
            Base state: "PLAN_EVASIVE_MANEUVER"
      - Elif p_collision >= 1e-5:
            Base state: "MONITOR_CLOSE_APPROACH"
      - Else:
            Base state: "MAINTAIN_COURSE"

    Fuel Constraint Override:
      - If base state is "PLAN_EVASIVE_MANEUVER" or "EMERGENCY_AVOIDANCE"
        AND fuel_remaining_delta_v <= 0:
            Return "CRITICAL_NO_FUEL_ALERT"

    Parameters:
        p_collision: Probability of collision (0.0 to 1.0).
        miss_distance_km: Minimum distance at closest approach in kilometers.
        fuel_remaining_delta_v: Available fuel for avoidance maneuver (delta-V units).

    Returns:
        str: Autonomous decision state string.
    """
    # 1. Determine base state from probability and geometric proximity
    if p_collision >= 1e-2 or miss_distance_km < 1.0:
        state = DecisionState.EMERGENCY_AVOIDANCE
    elif p_collision >= 1e-4:
        state = DecisionState.PLAN_EVASIVE_MANEUVER
    elif p_collision >= 1e-5:
        state = DecisionState.MONITOR_CLOSE_APPROACH
    else:
        state = DecisionState.MAINTAIN_COURSE

    # 2. Fuel constraint override rule
    if state in (DecisionState.PLAN_EVASIVE_MANEUVER, DecisionState.EMERGENCY_AVOIDANCE):
        if fuel_remaining_delta_v <= 0.0:
            return DecisionState.CRITICAL_NO_FUEL_ALERT.value

    return state.value


class DecisionEngine:
    """
    Class wrapper for the Decision Engine to support stateful configurations
    and backward compatibility with existing simulator modules.
    """
    def __init__(
        self,
        yellow_threshold: float = 1e-4,
        red_threshold: float = 1e-2,
        proximity_threshold_km: float = 1.0
    ):
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold
        self.proximity_threshold_km = proximity_threshold_km

    def evaluate_risk(
        self,
        p_collision: float,
        remaining_fuel: float,
        miss_distance_km: Optional[float] = None
    ) -> DecisionState:
        """
        Determines autonomous satellite action state.
        Supports both 2-argument (legacy) and 3-argument calls.
        """
        dist_km = miss_distance_km if miss_distance_km is not None else (0.5 if p_collision >= self.red_threshold else 100.0)
        state_str = evaluate_risk(p_collision, dist_km, remaining_fuel)
        return DecisionState(state_str)
