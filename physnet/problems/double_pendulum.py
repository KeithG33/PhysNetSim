# src/physnet/problems/pendulum.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence


# [theta1, theta2, omega1, omega2]
PendulumState = np.ndarray[float, 4]  


@dataclass
class PendulumParams:
    L1: float = 1.0
    L2: float = 1.0
    M1: float = 1.0
    M2: float = 1.0
    g:  float = 9.81


class DoublePendulum:
    """
    Double pendulum packaged to satisfy PhysProblem:
      - state (np.ndarray): [theta1, theta2, omega1, omega2] 
      - dt: float
      - derivs(t, y) -> dy/dt (np.ndarray)
      - init_draw / update_draw for Animator
    """
    def __init__(
        self,
        params: Optional[PendulumParams] = None,
        dt:     float = 0.05,
        state0: Optional[PendulumState] = None,
    ):
        params = params or PendulumParams()
        self.dt = dt
        self.state = np.array(state0 or [np.pi/4, np.pi/6, 0.0, 0.0], dtype=float)
        self.L1, self.L2, self.M1, self.M2, self.G = (
            params.L1, params.L2, params.M1, params.M2, params.g
        )    

    def derivs(self, t, y):
        """
        The system of ODEs for the double pendulum
        """
        theta1, theta2, omega1, omega2 = y

        dtheta = theta1 - theta2
        sin = np.sin
        cos = np.cos

        theta1_dot_num = self.M2 * self.L1 * omega1 ** 2 * sin(2 * dtheta) + 2 * self.M2 * self.L2 * omega2 ** 2 * sin(dtheta) + 2 * self.G * self.M2 * cos(theta2) * sin(dtheta) + 2 * self.G * self.M1 * sin(theta1)
        theta2_dot_num = self.M2 * self.L2 * omega2 ** 2 * sin(2 * dtheta) + 2 * (self.M1 + self.M2) * self.L1 * omega1 ** 2 * sin(dtheta) + 2 * self.G * (self.M1 + self.M2) * cos(theta1) * sin(dtheta)
        theta1_dot_denom = -2 * self.L1 * (self.M1 + self.M2 * sin(dtheta) ** 2)
        theta2_dot_denom = 2 * self.L2 * (self.M1 + self.M2 * sin(dtheta) ** 2)
        
        omega1_dot = theta1_dot_num / theta1_dot_denom
        omega2_dot = theta2_dot_num / theta2_dot_denom
    
        return np.array([omega1, omega2, omega1_dot, omega2_dot])

    def init_draw(self, ax: plt.Axes) -> Sequence:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        line, = ax.plot([], [], "o-", lw=2)
        return (line,)

    def update_draw(self, artists: Sequence) -> Sequence:
        line, = artists
        theta1, theta2, _, _ = self.state
        x1, y1 = np.sin(theta1), -np.cos(theta1)
        x2, y2 = x1 + np.sin(theta2), y1 - np.cos(theta2)
        line.set_data([0, x1, x2], [0, y1, y2])
        return (line,)
