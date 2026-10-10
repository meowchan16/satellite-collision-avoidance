"""
Conjunction Assessment Module.
Identifies orbital close encounters, determines the Time of Closest Approach (TCA),
and evaluates miss distances and relative velocity vectors between primary satellites
and tracked space debris objects.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Union
import numpy as np


def _ensure_utc(dt: datetime) -> datetime:
    """Ensures that a datetime instance is timezone-aware and set to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def find_tca(
    sat_obj: Any,
    debris_obj: Any,
    start_time: datetime,
    window_minutes: float = 90.0,
    coarse_step_seconds: float = 10.0,
    fine_step_seconds: float = 1.0,
) -> Dict[str, Any]:
    """
    Finds the Time of Closest Approach (TCA) between a satellite and a space debris object.

    Uses a hierarchical search strategy:
      1. Coarse search: Evaluates relative distance in steps of `coarse_step_seconds` (default 10s)
         over the screening window to locate the global/local minimum distance candidate.
      2. Fine search: Scans the neighborhood surrounding the coarse minimum in steps of
         `fine_step_seconds` (default 1s) to pinpoint the precise second of closest approach.

    Parameters:
        sat_obj: Primary space object (instance of SpaceObject or duck-typed with propagate_sgp4 / tle).
        debris_obj: Secondary space object (debris fragment with propagate_sgp4 / tle).
        start_time: Starting epoch datetime for the screening window.
        window_minutes: Duration of screening window in minutes (default: 90.0 minutes / ~1 LEO orbit).
        coarse_step_seconds: Time step in seconds for coarse pass (default: 10.0s).
        fine_step_seconds: Time step in seconds for fine pass (default: 1.0s).

    Returns:
        dict: A dictionary containing:
            - 'tca_time' (datetime): Precise timestamp of closest approach (UTC).
            - 'miss_distance_km' (float): Minimum 3D Euclidean distance between objects at TCA (km).
            - 'relative_velocity_km_s' (float): Magnitude of relative velocity vector at TCA (km/s).

    Raises:
        ValueError: If start_time or window_minutes are invalid.
    """
    if window_minutes <= 0:
        raise ValueError(f"window_minutes must be positive, got {window_minutes}")

    current_time = _ensure_utc(start_time)
    total_window_seconds = int(window_minutes * 60.0)

    # ------------------------------------------------------------------
    # Step 1: Coarse Grid Search
    # ------------------------------------------------------------------
    coarse_times = [
        current_time + timedelta(seconds=sec)
        for sec in range(0, total_window_seconds + 1, int(coarse_step_seconds))
    ]

    min_coarse_dist = float("inf")
    best_coarse_time = current_time

    for t in coarse_times:
        r_sat, _ = sat_obj.propagate_sgp4(t)
        r_deb, _ = debris_obj.propagate_sgp4(t)
        dist = float(np.linalg.norm(r_sat - r_deb))

        if dist < min_coarse_dist:
            min_coarse_dist = dist
            best_coarse_time = t

    # ------------------------------------------------------------------
    # Step 2: Fine Search Around Coarse Minimum
    # ------------------------------------------------------------------
    # Window spans [-coarse_step_seconds, +coarse_step_seconds] around best coarse time
    fine_start_sec = max(0, int((best_coarse_time - current_time).total_seconds()) - int(coarse_step_seconds))
    fine_end_sec = min(total_window_seconds, int((best_coarse_time - current_time).total_seconds()) + int(coarse_step_seconds))

    fine_times = [
        current_time + timedelta(seconds=sec)
        for sec in range(fine_start_sec, fine_end_sec + 1, int(fine_step_seconds))
    ]

    tca_time = best_coarse_time
    min_dist = float("inf")
    best_v_rel_mag = 0.0

    for t in fine_times:
        r_sat, v_sat = sat_obj.propagate_sgp4(t)
        r_deb, v_deb = debris_obj.propagate_sgp4(t)
        dist = float(np.linalg.norm(r_sat - r_deb))

        if dist < min_dist:
            min_dist = dist
            tca_time = t
            # Relative velocity vector: v_rel = v_sat - v_deb
            v_rel = v_sat - v_deb
            best_v_rel_mag = float(np.linalg.norm(v_rel))

    return {
        "tca_time": tca_time,
        "miss_distance_km": float(min_dist),
        "relative_velocity_km_s": float(best_v_rel_mag),
    }
