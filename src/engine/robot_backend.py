from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class FKResult:
    position: np.ndarray  # (x, y)
    yaw: float            # radians
    singular: bool


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    reason: str


class Planar3RRobot:
    """Planar 3R (Z-axis revolute) robot using pure NumPy.

    Link lengths are in meters. Motion constrained to XY plane.
    """

    def __init__(self, link_lengths: List[float]):
        self.link_lengths = list(link_lengths)

    def scale_links(self, factor: float) -> None:
        self.link_lengths = [max(0.01, l * factor) for l in self.link_lengths]

    # ------------- Kinematics -------------
    def fk(self, q: np.ndarray) -> FKResult:
        l1, l2, l3 = self.link_lengths
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        a1 = q1
        a2 = q1 + q2
        a3 = q1 + q2 + q3
        x = l1 * np.cos(a1) + l2 * np.cos(a2) + l3 * np.cos(a3)
        y = l1 * np.sin(a1) + l2 * np.sin(a2) + l3 * np.sin(a3)
        yaw = a3
        singular = self._is_singular(q, solve_yaw=False)
        return FKResult(position=np.array([x, y], dtype=float), yaw=float(yaw), singular=singular)

    def _jacobian_xy(self, q: np.ndarray) -> np.ndarray:
        l1, l2, l3 = self.link_lengths
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        a1 = q1
        a2 = q1 + q2
        a3 = q1 + q2 + q3

        # Partial derivatives for planar 3R
        dxdq1 = -l1 * np.sin(a1) - l2 * np.sin(a2) - l3 * np.sin(a3)
        dxdq2 = -l2 * np.sin(a2) - l3 * np.sin(a3)
        dxdq3 = -l3 * np.sin(a3)
        dydq1 =  l1 * np.cos(a1) + l2 * np.cos(a2) + l3 * np.cos(a3)
        dydq2 =  l2 * np.cos(a2) + l3 * np.cos(a3)
        dydq3 =  l3 * np.cos(a3)
        J = np.array([
            [dxdq1, dxdq2, dxdq3],
            [dydq1, dydq2, dydq3],
        ], dtype=float)
        return J

    def jacobian_xy(self, q: np.ndarray) -> np.ndarray:
        """Public accessor for the 2x3 position Jacobian."""
        return self._jacobian_xy(q)

    def _is_singular(self, q: np.ndarray, solve_yaw: bool = False) -> bool:
        if solve_yaw:
            # Combine [x,y] with yaw derivative [1,1,1]
            J_xy = self._jacobian_xy(q)
            J = np.vstack([J_xy, np.array([[1.0, 1.0, 1.0]])])
            return np.linalg.matrix_rank(J) < 3
        else:
            J = self._jacobian_xy(q)
            # Robust numeric: smallest singular value threshold
            s = np.linalg.svd(J, compute_uv=False)
            sigma_min = float(s[-1]) if s.size > 0 else 0.0
            return sigma_min < 1e-3

    def singularity_metrics(self, q: np.ndarray) -> tuple[float, float]:
        """
        Return (sigma_min, cond) for the position Jacobian J (2x3).
        cond uses 2-norm via singular values. If sigma_min ~ 0 => singular.
        """
        J = self._jacobian_xy(q)
        s = np.linalg.svd(J, compute_uv=False)
        if s.size == 0:
            return 0.0, float("inf")
        sigma_min = float(s[-1])
        sigma_max = float(s[0])
        cond = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
        return sigma_min, cond

    def svd_jacobian(self, q: np.ndarray):
        """
        Return (U, S, Vt) SVD of J_xy where J = U diag(S) Vt.
        U is 2x2 (task-space directions), S length-2, Vt 3x3.
        """
        J = self._jacobian_xy(q)
        U, S, Vt = np.linalg.svd(J, full_matrices=True)
        return U, S, Vt

    def ik(self, position_xy: np.ndarray, yaw: Optional[float], q_init: Optional[np.ndarray] = None) -> IKResult:
        """Damped least-squares IK for position-only by default.

        If yaw is provided, it adds a soft yaw term, but primary task is (x,y).
        If q_init is provided, uses it as initial guess.
        """
        target = np.array([float(position_xy[0]), float(position_xy[1])], dtype=float)
        q = np.array(q_init, dtype=float) if q_init is not None else np.zeros(3, dtype=float)
        lam = 1e-2
        success = False
        reason = "max_iter"
        for _ in range(300):
            fk = self.fk(q)
            e_xy = target - fk.position  # 2-vector
            if np.linalg.norm(e_xy) < 1e-4:
                success = True
                reason = "converged"
                break
            J = self._jacobian_xy(q)  # 2x3
            # DLS step
            H = J.T @ J + (lam ** 2) * np.eye(3)
            dq = np.linalg.solve(H, J.T @ e_xy)
            # limit step to avoid instability
            step = np.clip(dq, -0.2, 0.2)
            q = q + step
            # normalize angles to [-pi, pi]
            q = (q + np.pi) % (2 * np.pi) - np.pi
        return IKResult(q=q, success=success, reason=reason)

    def _ik_2r(self, target_xy: np.ndarray, l1: float, l2: float, elbow_up: bool = True) -> Optional[np.ndarray]:
        """Analytical IK for 2R planar robot.
        
        Returns [q1, q2] or None if unreachable.
        elbow_up: True for elbow up configuration, False for elbow down.
        """
        x, y = float(target_xy[0]), float(target_xy[1])
        r = np.hypot(x, y)
        
        # Check reachability
        if r > l1 + l2 or r < abs(l1 - l2):
            return None
        
        # Calculate q2 using cosine law
        cos_q2 = (l1*l1 + l2*l2 - r*r) / (2 * l1 * l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        q2 = np.arccos(cos_q2)
        
        # Choose elbow configuration
        if not elbow_up:
            q2 = -q2
        
        # Calculate q1
        alpha = np.arctan2(y, x)
        beta = np.arccos(np.clip((l1*l1 + r*r - l2*l2) / (2 * l1 * r), -1.0, 1.0))
        q1 = alpha - beta if elbow_up else alpha + beta
        
        return np.array([q1, q2], dtype=float)
    
    def ik_flip_configuration(self, position_xy: np.ndarray, q_current: np.ndarray, prefer_elbow_up: Optional[bool] = None) -> IKResult:
        """Find alternative IK configuration (elbow up/down flip) for the same end-effector position.
        
        Tries to find opposite elbow configuration by using iterative IK with 
        initial guesses that promote the opposite configuration.
        
        Args:
            prefer_elbow_up: If True, prefers elbow up; if False, prefers elbow down.
                            If None, finds opposite of current configuration.
        """
        target = np.array([float(position_xy[0]), float(position_xy[1])], dtype=float)
        q2_curr = float(q_current[1])
        
        # Determine current elbow configuration (rough: positive q2 ≈ elbow up)
        current_elbow_up = q2_curr > -0.1
        
        # Determine target elbow configuration
        if prefer_elbow_up is None:
            # Find opposite of current
            target_elbow_up = not current_elbow_up
        else:
            target_elbow_up = prefer_elbow_up
        
        # Try to find alternative configuration with target elbow
        best_result = None
        best_error = float('inf')
        
        # Try multiple initial guesses that encourage opposite elbow
        # Flip q2 and vary q1, q3 to explore alternative solutions
        for q1_offset in np.linspace(-np.pi/2, np.pi/2, 5):
                    for q2_magnitude in [0.3, 0.5, 0.7, 1.0, 1.5]:
                        for q3_offset in np.linspace(-np.pi/2, np.pi/2, 5):
                            q_guess = q_current.copy()
                            q_guess[0] += q1_offset
                            # Set q2 to target elbow configuration
                            q_guess[1] = q2_magnitude if target_elbow_up else -q2_magnitude
                            q_guess[2] += q3_offset
                            q_guess = (q_guess + np.pi) % (2 * np.pi) - np.pi
                            
                            result = self.ik(target, None, q_init=q_guess)
                            if result.success:
                                fk_check = self.fk(result.q)
                                error = np.linalg.norm(fk_check.position - target)
                                
                                # Check if this matches target configuration
                                result_q2 = result.q[1]
                                matches_target = False
                                if target_elbow_up and result_q2 > 0.05:
                                    matches_target = True
                                elif not target_elbow_up and result_q2 < -0.05:
                                    matches_target = True
                                
                                # Also check if different from current (unless same as current)
                                is_different = (target_elbow_up != current_elbow_up)
                                
                                if (matches_target or is_different) and error < best_error:
                                    best_error = error
                                    best_result = result
                                    
                                    if best_error < 1e-4:
                                        return best_result
        
        # If found good alternative, return it
        if best_result is not None and best_error < 1e-3:
            return best_result
        
        # Fallback: try with simple q2 set to target configuration
        q_simple = q_current.copy()
        if target_elbow_up:
            q_simple[1] = abs(q_simple[1]) if abs(q_simple[1]) > 0.1 else 0.5
        else:
            q_simple[1] = -abs(q_simple[1]) if abs(q_simple[1]) > 0.1 else -0.5
        return self.ik(target, None, q_init=q_simple)

    # ------------- Geometry helpers -------------
    def link_endpoints(self, q: np.ndarray) -> np.ndarray:
        """Return points [p0, p1, p2, p3] in world frame (XY)."""
        l1, l2, l3 = self.link_lengths
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        p0 = np.array([0.0, 0.0], dtype=float)
        p1 = np.array([l1 * np.cos(q1), l1 * np.sin(q1)], dtype=float)
        a2 = q1 + q2
        p2 = p1 + np.array([l2 * np.cos(a2), l2 * np.sin(a2)], dtype=float)
        a3 = a2 + q3
        p3 = p2 + np.array([l3 * np.cos(a3), l3 * np.sin(a3)], dtype=float)
        return np.vstack([p0, p1, p2, p3])

    # ------------- Reachability -------------
    def within_reach(self, point_xy: np.ndarray) -> bool:
        r = float(np.hypot(point_xy[0], point_xy[1]))
        L = self.link_lengths
        r_max = sum(L)
        r_min = max(0.0, abs(L[0] - (L[1] + L[2])))
        return r_min - 1e-6 <= r <= r_max + 1e-6


