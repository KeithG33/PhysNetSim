# examples/run_double_pendulum.py
from __future__ import annotations
import argparse
import numpy as np

from physnet.simulator import Animator
from physnet.problems.double_pendulum import DoublePendulum, PendulumParams


def main():
    p = argparse.ArgumentParser(description="Run a double pendulum demo (angles in DEGREES).")
   
    # physical params
    p.add_argument("--L1", type=float, default=1.0, help="length of rod 1 (m)")
    p.add_argument("--L2", type=float, default=1.0, help="length of rod 2 (m)")
    p.add_argument("--M1", type=float, default=1.0, help="mass 1 (kg)")
    p.add_argument("--M2", type=float, default=1.0, help="mass 2 (kg)")
    p.add_argument("--g",  type=float, default=9.81, help="gravity (m/s^2)")

    # initial conditions (input angles in degrees)
    p.add_argument("--theta1", type=float, default=45.0, help="initial θ1 in degrees")
    p.add_argument("--theta2", type=float, default=30.0, help="initial θ2 in degrees")
    p.add_argument("--omega1", type=float, default=0.0, help="initial ω1 (rad/s)")
    p.add_argument("--omega2", type=float, default=0.0, help="initial ω2 (rad/s)")
    
    # integration + animation
    p.add_argument("--dt", type=float, default=0.02, help="time step (s)")
    p.add_argument("--frames", type=int, default=600, help="number of frames")

    args = p.parse_args()

    # Convert user angles (degrees) → radians for the physics
    th1 = np.deg2rad(args.theta1)
    th2 = np.deg2rad(args.theta2)

    params = PendulumParams(L1=args.L1, L2=args.L2, M1=args.M1, M2=args.M2, g=args.g)
    system = DoublePendulum(params=params, dt=args.dt, state0=(th1, th2, args.omega1, args.omega2))

    Animator(system, frames=args.frames).animate()

if __name__ == "__main__":
    main()
