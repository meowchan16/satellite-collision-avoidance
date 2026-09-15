"""
Orbital Physics and Kinematics Engine.
Integrates SGP4 orbital propagation, unified 3D Cartesian state vectors r=[x, y, z] (km),
v=[vx, vy, vz] (km/s), and sensor measurement noise layer for real-world tracking uncertainty.
"""

from typing import List, Optional, Tuple, Union
from datetime import datetime, timezone
import numpy as np
from sgp4.api import Satrec, WGS72, jday

# Backward compatibility & integration with data_ingestion
from core.data_ingestion import TLEObject


class SpaceObject:
    """
    Represents a satellite or debris object in space.
    Supports both direct Cartesian kinematic motion and high-fidelity SGP4 orbital propagation.
    """
    def __init__(
        self,
        name: str,
        position: Union[np.ndarray, List[float]],
        velocity: Union[np.ndarray, List[float]],
        is_satellite: bool = False,
        norad_id: Optional[int] = None,
        object_type: str = "UNKNOWN",
        orbit_class: str = "UNKNOWN",
        tle_line1: Optional[str] = None,
        tle_line2: Optional[str] = None,
        epoch: Optional[datetime] = None
    ):
        self.name = name
        self.position = np.array(position, dtype=float)  # [x, y, z] in km (inertial TEME or local Cartesian)
        self.velocity = np.array(velocity, dtype=float)  # [vx, vy, vz] in km/s
        self.is_satellite = is_satellite
        self.norad_id = norad_id
        self.object_type = object_type if object_type != "UNKNOWN" else ("PAYLOAD" if is_satellite else "DEBRIS")
        self.orbit_class = orbit_class
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.epoch = epoch or datetime.now(timezone.utc)
        self.trajectory_history = [self.position.copy()]

        # Initialize SGP4 satellite record if TLE strings are supplied
        self.satrec: Optional[Satrec] = None
        if self.tle_line1 and self.tle_line2:
            self._init_sgp4()

    def _init_sgp4(self) -> None:
        """Initializes Satrec record from TLE lines."""
        if not (self.tle_line1 and self.tle_line2):
            return
        self.satrec = Satrec.twoline2rv(self.tle_line1, self.tle_line2, WGS72)
        if self.norad_id is None and hasattr(self.satrec, "satnum"):
            self.norad_id = self.satrec.satnum

    @classmethod
    def from_tle(cls, tle_obj: TLEObject, is_satellite: Optional[bool] = None, epoch: Optional[datetime] = None) -> "SpaceObject":
        """Factory method to construct a SpaceObject directly from a parsed TLEObject."""
        sat = cls(
            name=tle_obj.name,
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            is_satellite=is_satellite if is_satellite is not None else (tle_obj.object_type.upper() == "PAYLOAD"),
            norad_id=tle_obj.norad_id,
            object_type=tle_obj.object_type,
            orbit_class=tle_obj.orbit_class,
            tle_line1=tle_obj.line1,
            tle_line2=tle_obj.line2,
            epoch=epoch
        )
        # Propagate to initial epoch
        eval_time = epoch or datetime.now(timezone.utc)
        sat.propagate_sgp4(eval_time)
        # Reset trajectory history to start with the propagated epoch position
        sat.trajectory_history = [sat.position.copy()]
        return sat

    def update_position(self, dt: float) -> np.ndarray:
        """Propagates state vector forward linearly by time step dt (seconds)."""
        self.position += self.velocity * dt
        self.trajectory_history.append(self.position.copy())
        return self.position

    def propagate_sgp4(self, target_time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates orbit state to target_time using SGP4.
        Returns:
            r: np.ndarray [x, y, z] in km (True Equator, Mean Equinoctial / TEME frame)
            v: np.ndarray [vx, vy, vz] in km/s
        Raises:
            RuntimeError: If SGP4 evaluation fails or TLE not present.
        """
        if self.satrec is None:
            raise ValueError(f"Object {self.name} does not have initialized SGP4 TLE records.")

        # Ensure UTC timezone
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        jd, fr = jday(
            target_time.year,
            target_time.month,
            target_time.day,
            target_time.hour,
            target_time.minute,
            target_time.second + target_time.microsecond * 1e-6
        )

        error_code, r_teme, v_teme = self.satrec.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(f"SGP4 error code {error_code} while propagating {self.name} at {target_time.isoformat()}")

        self.position = np.array(r_teme, dtype=float)
        self.velocity = np.array(v_teme, dtype=float)
        self.epoch = target_time
        self.trajectory_history.append(self.position.copy())
        return self.position, self.velocity

    def get_noisy_observation(
        self,
        sigma_pos: float = 0.5,
        sigma_vel: float = 0.001,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the Sensor Noise Layer.
        Models real-world tracking uncertainty by injecting zero-mean Gaussian noise N(0, sigma^2)
        into the true position and velocity states.

        Parameters:
            sigma_pos: Position 1-sigma tracking error in km (default 0.5 km)
            sigma_vel: Velocity 1-sigma tracking error in km/s (default 0.001 km/s = 1 m/s)
            rng: Optional numpy random Generator for deterministic reproducibility

        Returns:
            noisy_pos: Observed [x, y, z] (km)
            noisy_vel: Observed [vx, vy, vz] (km/s)
        """
        generator = rng if rng is not None else np.random.default_rng()
        pos_noise = generator.normal(0.0, sigma_pos, size=3)
        vel_noise = generator.normal(0.0, sigma_vel, size=3)
        return self.position + pos_noise, self.velocity + vel_noise


class PhysicsEngine:
    """
    Astrodynamics & Kinematics propagation engine.
    Supports constant velocity linear kinematic stepping as well as batch SGP4 propagation.
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt

    def step(self, space_objects: List[SpaceObject]) -> None:
        """Updates positions of all space objects by one dt time step."""
        for obj in space_objects:
            obj.update_position(self.dt)

    def propagate_all_sgp4(
        self,
        space_objects: List[SpaceObject],
        target_time: datetime
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Propagates all TLE-backed space objects in the catalog to target_time via SGP4.
        Returns a list of (position, velocity) pairs.
        """
        results = []
        for obj in space_objects:
            if obj.satrec is not None:
                r, v = obj.propagate_sgp4(target_time)
                results.append((r, v))
            else:
                # Fallback to linear step for non-TLE objects
                obj.update_position(self.dt)
                results.append((obj.position, obj.velocity))
        return results

    @staticmethod
    def apply_sensor_noise_layer(
        space_objects: List[SpaceObject],
        sigma_pos: float = 0.5,
        sigma_vel: float = 0.001,
        rng: Optional[np.random.Generator] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Applies the sensor noise layer across an ensemble of space objects.
        Returns a list of observed (noisy_r, noisy_v) tuples.
        """
        generator = rng if rng is not None else np.random.default_rng()
        return [obj.get_noisy_observation(sigma_pos=sigma_pos, sigma_vel=sigma_vel, rng=generator) for obj in space_objects]
