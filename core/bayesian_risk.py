"""
Probabilistic Collision Risk Engine.
Models sensor measurement noise propagation and computes continuous collision
probability (P_c) between active spacecraft and space debris using standard
conjunction assessment formulas (Foster & Akella / Chan 2D Gaussian B-plane PDF approximation).
"""

from typing import Union, Optional
import numpy as np


def calculate_collision_probability(
    miss_distance_km: float,
    sigma_km: float,
    hard_body_radius_m: float = 3.0
) -> float:
    """
    Computes the probability of collision (P_c) using the standard simplified 2D Gaussian
    Probability Density Function (PDF) approximation for orbital conjunction assessment:

        P_c = (R^2 / (2 * sigma^2)) * exp(- (d^2) / (2 * sigma^2))

    Mathematical Rationale:
      - In orbital conjunction analysis (Akella & Alfriend / Foster model), when two objects
        encounter each other at high relative velocity, the collision problem is projected onto
        the 2D encounter plane (B-plane) perpendicular to the relative velocity vector.
      - Assuming symmetric isotropic positional uncertainty sigma (km) and small hard-body radius R (km)
        relative to the position uncertainty (R << sigma), the circular integral of the 2D Gaussian PDF
        over the collision cross-section area pi * R^2 simplifies to:
            P_c approx [Area / (2 * pi * sigma^2)] * exp(- d^2 / (2 * sigma^2))
                 = [(pi * R^2) / (2 * pi * sigma^2)] * exp(- d^2 / (2 * sigma^2))
                 = (R^2 / (2 * sigma^2)) * exp(- d^2 / (2 * sigma^2))
      - d: Miss distance at TCA (km).
      - R: Combined hard-body radius (km) = (R_sat + R_debris).
      - sigma: Combined standard deviation of tracking uncertainty (km).

    Parameters:
        miss_distance_km: Distance between objects at Time of Closest Approach (km).
        sigma_km: Combined tracking position uncertainty standard deviation (km).
        hard_body_radius_m: Combined physical collision radius in meters (default: 3.0 m).

    Returns:
        float: Clamped probability of collision P_c in the range [0.0, 1.0].
    """
    # Guard against invalid or non-physical parameters
    if sigma_km <= 0.0 or np.isnan(sigma_km) or np.isinf(sigma_km):
        raise ValueError(f"sigma_km must be strictly positive and finite, got {sigma_km}")

    if miss_distance_km < 0.0 or np.isnan(miss_distance_km):
        raise ValueError(f"miss_distance_km must be non-negative and finite, got {miss_distance_km}")

    if hard_body_radius_m <= 0.0 or np.isnan(hard_body_radius_m):
        return 0.0

    # 1. Convert hard-body radius from meters to kilometers
    # 1 km = 1000 m
    radius_km = hard_body_radius_m / 1000.0

    # 2. Compute variance terms
    two_sigma_sq = 2.0 * (sigma_km ** 2)

    # 3. Geometric scaling factor: (R^2 / (2 * sigma^2))
    scale_factor = (radius_km ** 2) / two_sigma_sq

    # 4. Exponential Gaussian distance attenuation: exp(- d^2 / (2 * sigma^2))
    exponent = - (miss_distance_km ** 2) / two_sigma_sq
    
    # Avoid numerical underflow for large miss distances (e.g., > 700 sigma)
    if exponent < -700.0:
        return 0.0

    prob = scale_factor * np.exp(exponent)

    # 5. Clamp probability between [0.0, 1.0]
    return float(np.clip(prob, 0.0, 1.0))


class BayesianRiskEngine:
    """
    Probabilistic risk estimation engine.
    Wraps calculate_collision_probability and provides backward compatibility
    for state vectors and Streamlit application interfaces.
    """
    def __init__(self, sensor_sigma_pos: float = 0.5):
        """
        Parameters:
            sensor_sigma_pos: Positional tracking uncertainty 1-sigma in km (default: 0.5 km).
        """
        self.sensor_sigma_pos = sensor_sigma_pos

    def calculate_collision_probability(
        self,
        sat_pos: Union[np.ndarray, float],
        debris_pos: Optional[Union[np.ndarray, float]] = None,
        hard_sphere_radius: float = 3.0
    ) -> float:
        """
        Computes continuous collision probability.
        Supports both:
          - (miss_distance_km, sigma_km) scalar call
          - (sat_pos, debris_pos) 3D coordinate vector call
        """
        # Case A: If sat_pos is a scalar (e.g. miss distance passed directly)
        if isinstance(sat_pos, (int, float)) and debris_pos is not None and isinstance(debris_pos, (int, float)):
            miss_distance = float(sat_pos)
            sigma = float(debris_pos)
            return calculate_collision_probability(miss_distance, sigma, hard_body_radius_m=hard_sphere_radius)

        # Case B: 3D vector positions
        if debris_pos is not None:
            sat_arr = np.asarray(sat_pos, dtype=float)
            deb_arr = np.asarray(debris_pos, dtype=float)
            distance = float(np.linalg.norm(sat_arr - deb_arr))
        else:
            distance = float(sat_pos)

        # Use engine's sensor_sigma_pos
        return calculate_collision_probability(
            miss_distance_km=distance,
            sigma_km=self.sensor_sigma_pos,
            hard_body_radius_m=hard_sphere_radius
        )
