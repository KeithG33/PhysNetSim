# examples/run_schrodinger_2d.py
from __future__ import annotations
import argparse
import matplotlib.pyplot as plt

from physnet.simulator import Animator
from physnet.problems.schrodinger_2d import Grid2D, Domain, Schrodinger2D


def main():
    p = argparse.ArgumentParser(description="2D Schrödinger equation demo (|psi|^2 surface).")
    p.add_argument("--nx", type=int, default=100, help="grid points in x")
    p.add_argument("--ny", type=int, default=100, help="grid points in y")
    p.add_argument("--Lx", type=float, default=1.0, help="domain length in x")
    p.add_argument("--Ly", type=float, default=1.0, help="domain length in y")
    p.add_argument("--dt", type=float, default=1e-5, help="time step")
    p.add_argument("--frames", type=int, default=400, help="number of frames")
    p.add_argument("--steps-per-frame", type=int, default=4, help="physics steps per animation frame")
    args = p.parse_args()

    # domain (grid + hard-wall BC)
    grid = Grid2D(nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly)
    dom = Domain(grid=grid)

    # physics system
    system = Schrodinger2D(dom, dt=args.dt, zmax=75.0)

    Animator(system, frames=args.frames, blit=False, steps_per_frame=args.steps_per_frame).animate()

if __name__ == "__main__":
    main()
