from __future__ import annotations

import itertools
import numpy as np

from .robot_backend import Planar3RRobot


class WorkspaceSampler:
    """Sample reachable (x, y) points by sweeping joint angles.

    This is a pragmatic sampler adequate for UI visualization.
    """

    def __init__(self, robot: Planar3RRobot):
        self.robot = robot

    def sample_points(self, samples_per_joint: int = 72) -> np.ndarray:
        q_lin = np.linspace(-np.pi, np.pi, samples_per_joint, endpoint=False)
        points = []
        for q1, q2, q3 in itertools.product(q_lin, repeat=3):
            fk = self.robot.fk(np.array([q1, q2, q3], dtype=float))
            points.append([fk.position[0], fk.position[1]])
        return np.asarray(points, dtype=float)


