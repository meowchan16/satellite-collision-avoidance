from enum import Enum

class DecisionState(Enum):
    MAINTAIN_COURSE = "MAINTAIN_COURSE"
    PLAN_EVASIVE_MANEUVER = "PLAN_EVASIVE_MANEUVER"
    EXECUTE_EVASIVE_MANEUVER = "EXECUTE_EVASIVE_MANEUVER"
    EMERGENCY_ABORT = "EMERGENCY_ABORT"

class DecisionEngine:
    """
    Rule-Based Decision Engine.
    Evaluates risk score against predefined safety thresholds and selects actions.
    """
    def __init__(self, yellow_threshold: float = 1e-4, red_threshold: float = 1e-2):
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold

    def evaluate_risk(self, p_collision: float, remaining_fuel: float) -> DecisionState:
        """
        Determines autonomous satellite action state.
        """
        if remaining_fuel <= 0 and p_collision >= self.red_threshold:
            return DecisionState.EMERGENCY_ABORT
        
        if p_collision >= self.red_threshold:
            return DecisionState.EXECUTE_EVASIVE_MANEUVER
        elif p_collision >= self.yellow_threshold:
            return DecisionState.PLAN_EVASIVE_MANEUVER
        else:
            return DecisionState.MAINTAIN_COURSE
