import numpy as np

class BayesianRiskEngine:
    """
    Probabilistic risk estimation engine.
    Models sensor measurement noise propagation and estimates continuous collision probability.
    """
    def __init__(self, sensor_sigma_pos: float = 0.5):
        self.sensor_sigma_pos = sensor_sigma_pos  # Positional uncertainty standard deviation (km)

    def calculate_collision_probability(self, sat_pos: np.ndarray, debris_pos: np.ndarray, hard_sphere_radius: float = 0.1) -> float:
        """
        Estimates continuous probability of collision (P_coll) based on positional distance
        and sensor noise covariance.
        """
        distance = np.linalg.norm(sat_pos - debris_pos)
        
        # Combined covariance matrix assuming independent spherical Gaussian noise
        sigma_combined = 2 * (self.sensor_sigma_pos ** 2)
        
        # Simplified continuous risk model based on 3D Gaussian PDF overlap
        prob = np.exp(- (distance ** 2) / (2 * sigma_combined))
        
        # Scale by hard sphere radius relative to noise scale
        scaled_p_coll = min(1.0, prob * (hard_sphere_radius / (self.sensor_sigma_pos + 1e-6)))
        return float(scaled_p_coll)
