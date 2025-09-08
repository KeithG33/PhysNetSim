# simulator.py

from __future__ import annotations
from typing import Any, Callable, Sequence, Protocol, Optional
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .solver import PhysProblem, PhysSolver, RK4Solver



# --- time evolution engine ---
@dataclass
class PhysSolverEngine:
    problem: PhysProblem
    stepper: PhysSolver = field(default_factory=RK4Solver)
    t: float = 0.0

    @property
    def state(self): return self.problem.state
    @property
    def dt(self): return self.problem.dt

    def step(self):
        self.problem = self.stepper.step(self.problem, self.t)  # returns problem (mutated)
        self.t += self.problem.dt
        return self.problem.state
    

# --------- Animator that takes a Problem ---------
class Animator:
    def __init__(
        self,
        problem: PhysProblem,
        *,
        frames: int = 300,
        interval_ms: int = 50,
        steps_per_frame: int = 1,
        stepper: Optional[PhysSolver] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        blit: bool = False,
    ):
        self.problem = problem
        self.engine = PhysSolverEngine(problem=problem, stepper=stepper or RK4Solver())
        self.frames = frames
        self.interval_ms = interval_ms
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.blit = blit

        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        elif fig is not None:
            self.fig = fig
            self.ax = fig.gca()  # get/create current Axes on the Figure
        else:
            self.fig, self.ax = plt.subplots()

        self._artists: Optional[Sequence] = None

    def _init(self):
        self._artists = tuple(self.problem.init_draw(self.ax))
        return self._artists

    def _update(self, _k):
        for _ in range(self.steps_per_frame):
            self.engine.step()
        return tuple(self.problem.update_draw(self._artists))

    def animate(self):
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            init_func=self._init,
            frames=self.frames,
            interval=self.interval_ms,
            blit=self.blit,
        )
        plt.show()
        return ani
