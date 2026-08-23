import numpy as np

class SpaceObject:
    """
    Represents a satellite or debris object in space.
    """
    def __init__(self, name: str, position: np.ndarray, velocity: np.ndarray, is_satellite: bool = False):
        self.name = name
        self.position = np.array(position, dtype=float)  # [x, y, z] in km or meters
        self.velocity = np.array(velocity, dtype=float)  # [vx, vy, vz] in km/s or m/s
        self.is_satellite = is_satellite
        self.trajectory_history = [self.position.copy()]

    def update_position(self, dt: float):
        """Propagates state vector forward by time step dt."""
        self.position += self.velocity * dt
        self.trajectory_history.append(self.position.copy())

class PhysicsEngine:
    """
    Kinematic & Orbital motion propagation engine.
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt

    def step(self, space_objects: list):
        """Updates positions of all space objects by one time step."""
        for obj in space_objects:
            obj.update_position(self.dt)
