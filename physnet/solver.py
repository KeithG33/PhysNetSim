from typing import Any, Protocol, Sequence
import matplotlib.pyplot as plt
import torch
import numpy as np


class PhysProblem(Protocol):
    """ Defines a physics situation, or "problem" in the educational sense.
      - holds mutable `state`
      - provides `dt` and `derivs(t, y)->dy/dt`
      - provides draw hooks used by Animator
    """
    state: np.ndarray
    dt: float
    def derivs(self, t: float, y: np.ndarray) -> np.ndarray: ...
    def init_draw(self, ax: plt.Axes) -> Sequence: ...
    def update_draw(self, artists: Sequence) -> Sequence: ...


# Generic solver interface. 
# Given a problem and current time, return problem at next time step.
class PhysSolver(Protocol):
    def step(self, problem: PhysProblem, t: float) -> PhysProblem: ...


class RK4Solver:
    """Calculate next step using RK4 method."""
    def RungeKutta4(self,t, y, dt, derivs):
        """
        RK4 integration of a system of ODEs
        """
        k1 = dt * derivs(t, y)
        k2 = dt * derivs(t + dt / 2, y + k1 / 2)
        k3 = dt * derivs(t + dt / 2, y + k2 / 2)
        k4 = dt * derivs(t + dt, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    def step(self, problem: PhysProblem, t: float) -> PhysProblem:
        y_next = self.RungeKutta4(t, problem.state, problem.dt, problem.derivs)
        problem.state = y_next
        return problem
   
    
class NeuralNetSolver:
    """Calculate next step using a neural network to predict derivatives."""
    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device

    def step(self, problem: PhysProblem, t: float) -> PhysProblem:
        state = torch.tensor(problem.state)
        with torch.inference_mode():
            y_next = self.model(state).cpu().numpy()
        problem.state = y_next
        return problem
