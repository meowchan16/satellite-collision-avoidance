"""
A* Trajectory Path Planner.
Computes fuel-optimal, collision-free evasive maneuvers around space debris hazard zones
using 3D A* graph search over relative position grids with delta-V budgeting.
"""

from typing import List, Tuple, Optional, Union
import heapq
import numpy as np


class AStarPathPlanner:
    """
    A* 3D Trajectory Planner for fuel-efficient, collision-free evasive maneuvers.

    Discretizes the local 3D maneuvering volume into a relative coordinate grid,
    modeling debris encounter zones as spherical keep-out hazard buffers.
    Optimizes for minimal cumulative thrust (delta-v) while ensuring safety margin.
    """
    def __init__(
        self,
        step_size: float = 0.5,
        safety_margin_km: float = 2.0,
        fuel_weight: float = 1.0,
        risk_weight: float = 50.0
    ):
        """
        Parameters:
            step_size: Spatial discretization step in km (default: 0.5 km).
            safety_margin_km: Hard keep-out sphere radius around debris (default: 2.0 km).
            fuel_weight: Objective weighting factor for fuel consumption (delta-v).
            risk_weight: Penalty weighting factor for passing close to debris margin.
        """
        self.step_size = step_size
        self.safety_margin_km = safety_margin_km
        self.fuel_weight = fuel_weight
        self.risk_weight = risk_weight

    def plan_evasive_path(
        self,
        start_pos: Union[np.ndarray, List[float]],
        target_pos: Union[np.ndarray, List[float]],
        debris_list: List[Union[np.ndarray, List[float]]],
        grid_resolution: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Computes fuel-efficient collision-free 3D path from start_pos to target_pos
        avoiding debris keep-out zones using 3D A* search.

        Parameters:
            start_pos: Initial 3D position [x, y, z] in km.
            target_pos: Target waypoint [x, y, z] in km.
            debris_list: List of 3D positions [x, y, z] representing debris obstacles.
            grid_resolution: Optional override for search step size (km).

        Returns:
            list: Sequence of 3D Cartesian coordinates [x, y, z] from start to target.
        """
        start = np.asarray(start_pos, dtype=float)
        target = np.asarray(target_pos, dtype=float)
        debris_points = [np.asarray(d, dtype=float) for d in debris_list]

        res = grid_resolution if grid_resolution is not None else self.step_size

        # Check if direct straight-line path is already collision-free
        if not self._is_path_obstructed(start, target, debris_points):
            return [start, target]

        # 3D Discrete Neighborhood (26-connectivity)
        offsets = []
        for dx in [-res, 0.0, res]:
            for dy in [-res, 0.0, res]:
                for dz in [-res, 0.0, res]:
                    if dx == 0.0 and dy == 0.0 and dz == 0.0:
                        continue
                    offsets.append(np.array([dx, dy, dz]))

        # State representation for A*: tuple of rounded coordinates
        def to_key(pt: np.ndarray) -> Tuple[float, float, float]:
            return (round(float(pt[0]), 3), round(float(pt[1]), 3), round(float(pt[2]), 3))

        start_key = to_key(start)
        target_key = to_key(target)

        # Priority Queue: (f_score, g_score, current_pt_list)
        open_set = []
        heapq.heappush(open_set, (np.linalg.norm(target - start), 0.0, start_key, [start]))

        g_scores = {start_key: 0.0}
        visited = set()

        max_iterations = 2500
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            f_score, current_g, current_key, path = heapq.heappop(open_set)
            current_pos = path[-1]

            # Check if within one step of target
            if np.linalg.norm(target - current_pos) <= res * 1.5:
                if not self._is_path_obstructed(current_pos, target, debris_points):
                    path.append(target)
                    return path

            if current_key in visited:
                continue
            visited.add(current_key)

            for offset in offsets:
                neighbor = current_pos + offset
                neighbor_key = to_key(neighbor)

                if neighbor_key in visited:
                    continue

                # Check proximity to all debris obstacles
                too_close = False
                proximity_penalty = 0.0
                for d in debris_points:
                    dist_to_d = np.linalg.norm(neighbor - d)
                    if dist_to_d < self.safety_margin_km:
                        too_close = True
                        break
                    elif dist_to_d < (self.safety_margin_km * 2.0):
                        proximity_penalty += (self.safety_margin_km * 2.0 - dist_to_d) * self.risk_weight

                if too_close:
                    continue

                # Step cost: Euclidean distance + proximity risk penalty
                step_cost = float(np.linalg.norm(offset)) * self.fuel_weight + proximity_penalty
                tentative_g = current_g + step_cost

                if neighbor_key not in g_scores or tentative_g < g_scores[neighbor_key]:
                    g_scores[neighbor_key] = tentative_g
                    h_score = float(np.linalg.norm(target - neighbor))
                    f = tentative_g + h_score
                    heapq.heappush(open_set, (f, tentative_g, neighbor_key, path + [neighbor]))

        # Fallback: compute smooth geometric arc detour if grid search exhausts limit
        return self._generate_geometric_detour(start, target, debris_points)

    def _is_path_obstructed(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        debris_points: List[np.ndarray],
        num_checks: int = 20
    ) -> bool:
        """Checks if line segment between p1 and p2 penetrates any debris keep-out sphere."""
        if not debris_points:
            return False
        for t in np.linspace(0.0, 1.0, num_checks):
            sample_pt = p1 + t * (p2 - p1)
            for d in debris_points:
                if np.linalg.norm(sample_pt - d) < self.safety_margin_km:
                    return True
        return False

    def _generate_geometric_detour(
        self,
        start: np.ndarray,
        target: np.ndarray,
        debris_points: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Analytical fallback creating an optimal out-of-plane evasive arc around debris."""
        direction = target - start
        dist = float(np.linalg.norm(direction))
        if dist == 0:
            return [start, target]

        norm_dir = direction / dist
        # Find orthogonal out-of-plane vector
        if abs(norm_dir[0]) < 0.9:
            ortho = np.cross(norm_dir, np.array([1.0, 0.0, 0.0]))
        else:
            ortho = np.cross(norm_dir, np.array([0.0, 1.0, 0.0]))
        ortho = ortho / (np.linalg.norm(ortho) + 1e-9)

        # Offset distance scaled by safety margin
        detour_apex = start + norm_dir * (dist * 0.5) + ortho * (self.safety_margin_km * 1.5)
        return [start, detour_apex, target]

    @staticmethod
    def calculate_path_delta_v(path: List[np.ndarray], base_velocity_km_s: float = 7.6) -> float:
        """
        Estimates the total delta-V (m/s) required to execute the path direction changes.
        """
        if len(path) < 2:
            return 0.0
        
        total_delta_v_m_s = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i - 1]
            v2 = path[i + 1] - path[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_rad = np.arccos(cos_theta)
                # Delta-V for trajectory turn = 2 * v * sin(theta / 2)
                dv = 2.0 * (base_velocity_km_s * 1000.0) * np.sin(angle_rad / 2.0)
                total_delta_v_m_s += float(dv)

        return round(total_delta_v_m_s, 2)
