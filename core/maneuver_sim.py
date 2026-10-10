"""
Orbital Maneuver Simulation Module.
Simulates impulsive orbital maneuvers (prograde, retrograde, radial, cross-track burns),
calculates post-burn state vectors, derives updated orbital parameters, and propagates
new perturbed trajectories.
"""

from typing import Tuple, Dict, Any, List, Union, Optional
from datetime import datetime, timedelta, timezone
import numpy as np

from core.physics import SpaceObject
from core.data_ingestion import TLEObject

# Standard Astrodynamics Gravitational Constant for Earth (km^3 / s^2)
MU_EARTH = 398600.4418
# Earth Mean Equator Radius (km)
R_EARTH = 6378.137


def execute_evasive_maneuver(
    current_pos: Union[np.ndarray, List[float]],
    current_vel: Union[np.ndarray, List[float]],
    delta_v_magnitude: float,
    burn_direction: str = "prograde"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes an impulsive orbital burn by modifying the spacecraft velocity vector.

    Parameters:
        current_pos: 3D position vector r = [x, y, z] in km.
        current_vel: 3D velocity vector v = [vx, vy, vz] in km/s.
        delta_v_magnitude: Delta-V burn magnitude in meters per second (m/s).
                           Positive values accelerate along the direction; negative decelerate.
        burn_direction: Burn orientation:
                        - 'prograde': along current velocity vector (raises apoapsis)
                        - 'retrograde': opposing velocity vector (lowers periapsis)
                        - 'radial_out': away from Earth center (perpendicular to velocity)
                        - 'radial_in': toward Earth center
                        - 'normal': out-of-plane cross-track (orbital plane change)

    Returns:
        new_velocity (np.ndarray): 3D velocity vector after burn in km/s.
        details (dict): Summary containing burn telemetry, post-burn speed,
                        estimated altitude (km), and updated orbit regime string.
    """
    r = np.asarray(current_pos, dtype=float)
    v = np.asarray(current_vel, dtype=float)

    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))

    if r_mag == 0.0 or v_mag == 0.0:
        raise ValueError("Position and velocity vectors must have non-zero magnitude.")

    # 1. Convert delta_v from m/s to km/s (1 m/s = 0.001 km/s)
    delta_v_km_s = float(delta_v_magnitude) / 1000.0

    # Unit vectors for Orbital Frame (VNB / RTN frame)
    v_unit = v / v_mag
    r_unit = r / r_mag
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    normal_unit = h_vec / h_mag if h_mag > 0 else np.array([0.0, 0.0, 1.0])
    radial_unit = np.cross(v_unit, normal_unit)

    dir_clean = burn_direction.lower().strip()

    if dir_clean == "prograde":
        burn_vec = v_unit * delta_v_km_s
        burn_label = f"{abs(delta_v_magnitude):.2f} m/s Prograde"
    elif dir_clean == "retrograde":
        burn_vec = -v_unit * abs(delta_v_km_s)
        burn_label = f"{abs(delta_v_magnitude):.2f} m/s Retrograde"
    elif dir_clean in ("radial_out", "radial"):
        burn_vec = r_unit * delta_v_km_s
        burn_label = f"{abs(delta_v_magnitude):.2f} m/s Radial-Out"
    elif dir_clean == "radial_in":
        burn_vec = -r_unit * abs(delta_v_km_s)
        burn_label = f"{abs(delta_v_magnitude):.2f} m/s Radial-In"
    elif dir_clean in ("normal", "cross_track"):
        burn_vec = normal_unit * delta_v_km_s
        burn_label = f"{abs(delta_v_magnitude):.2f} m/s Cross-Track Normal"
    else:
        burn_vec = v_unit * delta_v_km_s
        burn_label = f"{abs(delta_v_magnitude):.2f} m/s Custom ({dir_clean})"

    # 2. Compute NEW velocity vector after impulsive burn
    new_v = v + burn_vec
    new_v_mag = float(np.linalg.norm(new_v))

    # 3. Calculate post-burn orbital elements (Vis-Viva energy equation)
    # Specific orbital energy: epsilon = (v^2 / 2) - (mu / r)
    specific_energy = (new_v_mag ** 2) / 2.0 - (MU_EARTH / r_mag)
    # Semi-major axis a = - mu / (2 * epsilon)
    if specific_energy < 0:
        semi_major_axis = - MU_EARTH / (2.0 * specific_energy)
        # Approximate average orbit altitude
        avg_altitude_km = semi_major_axis - R_EARTH
    else:
        # Hyperbolic / escape trajectory
        avg_altitude_km = r_mag - R_EARTH

    curr_alt_km = r_mag - R_EARTH
    orbit_label = f"LEO {int(round(max(0.0, avg_altitude_km)))}km" if avg_altitude_km < 2000 else f"MEO {int(round(avg_altitude_km))}km"

    details = {
        "burn_executed": f"Burn executed: {burn_label}",
        "delta_v_m_s": float(delta_v_magnitude),
        "pre_burn_speed_km_s": round(v_mag, 4),
        "post_burn_speed_km_s": round(new_v_mag, 4),
        "current_altitude_km": round(curr_alt_km, 2),
        "post_burn_semi_major_axis_km": round(semi_major_axis if specific_energy < 0 else r_mag, 2),
        "new_orbit": orbit_label,
    }

    return new_v, details


def calculate_new_trajectory(
    tle: Union[Tuple[str, str], TLEObject, SpaceObject],
    maneuver_time: datetime,
    delta_v: float,
    burn_direction: str = "prograde",
    duration_minutes: float = 90.0,
    step_seconds: float = 30.0
) -> List[Tuple[float, float, float]]:
    """
    Computes a sequence of 3D (x, y, z) Cartesian points representing the spacecraft's
    new post-burn orbit trajectory using numerical orbital dynamics.

    Parameters:
        tle: TLE lines tuple (line1, line2), TLEObject, or SpaceObject instance.
        maneuver_time: Datetime epoch of burn execution.
        delta_v: Burn magnitude in m/s.
        burn_direction: Direction of burn ('prograde', 'radial_out', etc.).
        duration_minutes: Orbit propagation horizon in minutes (default: 90.0 min).
        step_seconds: Numerical trajectory point step size in seconds (default: 30.0s).

    Returns:
        list of (x, y, z) tuples in km.
    """
    if maneuver_time.tzinfo is None:
        maneuver_time = maneuver_time.replace(tzinfo=timezone.utc)

    # Resolve SpaceObject
    if isinstance(tle, SpaceObject):
        sat = tle
    elif isinstance(tle, (tuple, list)) and len(tle) >= 2:
        sat = SpaceObject("ManeuverSat", [0, 0, 0], [0, 0, 0], tle_line1=tle[0], tle_line2=tle[1])
    elif hasattr(tle, "line1") and hasattr(tle, "line2"):
        sat = SpaceObject(tle.name, [0, 0, 0], [0, 0, 0], tle_line1=tle.line1, tle_line2=tle.line2)
    else:
        raise ValueError("Invalid TLE input provided for trajectory calculation.")

    # 1. Propagate to maneuver epoch to find exact state vector at burn
    r_burn, v_burn = sat.propagate_sgp4(maneuver_time)

    # 2. Apply impulsive delta-V burn
    new_v_burn, _ = execute_evasive_maneuver(r_burn, v_burn, delta_v, burn_direction=burn_direction)

    # 3. Propagate forward using 2-body Keplerian + J2 numerical integration
    total_steps = int((duration_minutes * 60.0) / step_seconds)
    dt = float(step_seconds)

    trajectory_points: List[Tuple[float, float, float]] = []
    current_r = r_burn.copy()
    current_v = new_v_burn.copy()

    trajectory_points.append((float(current_r[0]), float(current_r[1]), float(current_r[2])))

    # RK4 Numerical Integration for Keplerian + Earth oblateness (J2) acceleration
    j2 = 1.08263e-3

    def gravitational_acc(r_vec: np.ndarray) -> np.ndarray:
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm == 0:
            return np.zeros(3)
        # Point mass gravity
        a_kepler = - (MU_EARTH / (r_norm ** 3)) * r_vec
        # J2 Earth oblateness perturbation
        z2 = r_vec[2] ** 2
        factor = 1.5 * j2 * MU_EARTH * (R_EARTH ** 2) / (r_norm ** 5)
        a_j2_x = factor * r_vec[0] * (5.0 * z2 / (r_norm ** 2) - 1.0)
        a_j2_y = factor * r_vec[1] * (5.0 * z2 / (r_norm ** 2) - 1.0)
        a_j2_z = factor * r_vec[2] * (5.0 * z2 / (r_norm ** 2) - 3.0)
        return a_kepler + np.array([a_j2_x, a_j2_y, a_j2_z])

    for _ in range(total_steps):
        # Runge-Kutta 4th order step
        k1_v = gravitational_acc(current_r)
        k1_r = current_v

        k2_v = gravitational_acc(current_r + 0.5 * dt * k1_r)
        k2_r = current_v + 0.5 * dt * k1_v

        k3_v = gravitational_acc(current_r + 0.5 * dt * k2_r)
        k3_r = current_v + 0.5 * dt * k2_v

        k4_v = gravitational_acc(current_r + dt * k3_r)
        k4_r = current_v + dt * k3_v

        current_r += (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        current_v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        trajectory_points.append((float(current_r[0]), float(current_r[1]), float(current_r[2])))

    return trajectory_points
