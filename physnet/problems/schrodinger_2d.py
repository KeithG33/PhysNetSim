# src/physnet/problems/tdse2d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


@dataclass
class Grid2D:
    nx: int
    ny: int
    Lx: float = 1.0
    Ly: float = 1.0
    def __post_init__(self):
        self.x = np.linspace(0.0, self.Lx, self.nx)
        self.y = np.linspace(0.0, self.Ly, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")


class DirichletBC:
    def apply_state(self, u2d: np.ndarray) -> None:
        u2d[0,:] = u2d[-1,:] = u2d[:,0] = u2d[:,-1] = 0.0
    def apply_rhs(self, f2d: np.ndarray) -> None:
        f2d[0,:] = f2d[-1,:] = f2d[:,0] = f2d[:,-1] = 0.0


@dataclass
class Domain:
    grid: Grid2D
    bc: DirichletBC = DirichletBC()


def laplacian5(u: Array, dx: float, dy: float) -> Array:
    """Second-order five-point Laplacian with Dirichlet edges."""
    lap = np.zeros_like(u, dtype=np.complex128)
    lap[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx*dx)
        + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy*dy)
    )
    return lap


def make_double_slit_potential(g: Grid2D) -> Array:
    """Horizontal barrier at y≈Ly/2 with two non-overlapping slits.

    Note: ensures the y-band captures at least one grid row even if
    `bwidth < dy`, which otherwise can produce an empty barrier mask.
    """
    V = np.zeros((g.nx, g.ny), float)
    V0 = 50000

    # Parameters (centered in the grid)
    blength = 0.8   # total length of barrier along x (longer)
    bwidth  = 0.001  # thickness of barrier along y
    sdistance = 0.08  # center-to-center separation between slits (along x)
    ssize   = 0.06  # slit width (along x) — wider openings for more transmission

    # Axis masks
    x = g.x; y = g.y
    half_y = max(0.5*bwidth, 0.5*g.dy)
    mask_y_band = (np.abs(y - 0.5*g.Ly) <= half_y)
    mask_x_bar  = (np.abs(x - 0.5*g.Lx) <= 0.5*blength)

    # Base barrier mask (rectangle)
    barrier_mask = mask_x_bar[:, None] & mask_y_band[None, :]
    V[barrier_mask] = V0

    # Carve two slits: centers separated by sdistance; non-overlapping by design
    c1 = 0.5*g.Lx - 0.5*sdistance
    c2 = 0.5*g.Lx + 0.5*sdistance
    slit1_x = (np.abs(x - c1) <= 0.5*ssize)
    slit2_x = (np.abs(x - c2) <= 0.5*ssize)
    slit_mask = (slit1_x | slit2_x)[:, None] & mask_y_band[None, :]
    V[slit_mask] = 0.0

    if not barrier_mask.any():
        print("Warning: double-slit barrier mask is empty; increase bwidth or resolution.")

    return V


def make_gaussian_packet(g: Grid2D) -> Array:
    """Normalized Gaussian centered below barrier and moving upward.

    Packet starts near the bottom (y0 ≈ 0.1 Ly) and travels toward +y
    so it interacts with the horizontal barrier at y ≈ 0.5 Ly.
    """
    X, Y = g.X, g.Y
    x0, y0 = 0.5*g.Lx, 0.1*g.Ly
    sx, sy = 0.15*g.Lx, 0.05*g.Ly
    kx, ky = 0.0, -500.0

    psi0 = np.exp(-0.5*((X-x0)/sx)**2 - 0.5*((Y-y0)/sy)**2) * np.exp(1j*(kx*X + ky*Y))
    norm = np.sqrt((np.abs(psi0)**2).sum() * g.dx * g.dy)
    return (psi0 / norm) if norm > 0 else psi0


# ---------------- TDSE (PhysProblem-compatible) ----------------
class Schrodinger2D:
    """ Problem: 2D time-dependent Schrödinger equation with double-slit potential. Following equation,
    
    i * dpsi/dt = -0.5 * (psi_xx + psi_yy) + V(x,y) * psi

    And containing a double slit potential with initial Gaussian wave packet

    """
    def __init__(self, domain: Domain, dt: float = 2e-4, *, zmax: float = 75.0):
        self.dom = domain
        self.grid = domain.grid
        self.bc = domain.bc

        self.dt = float(dt)
        self._steps = 0
        self.renormalize_every = 20  # small fix for explicit RK drift
        self._zmax_fixed = float(zmax)

        # setup double slit + boosted Gaussian
        psi0 = make_gaussian_packet(self.grid)
        self.V  = make_double_slit_potential(self.grid)

        self.state = psi0.astype(np.complex128).reshape(-1)
        self._surf = None # for drawing
        self._ax3d = None # for drawing

    def derivs(self, t: float, y_flat: Array) -> Array:
        # Avoid in-place mutation of the integrator's input state
        g = self.grid
        psi = y_flat.reshape(g.nx, g.ny)
        # work buffer
        if not hasattr(self, "_psi_bc"):
            self._psi_bc = np.empty_like(psi)
        self._psi_bc[...] = psi
        self.bc.apply_state(self._psi_bc)

        Hpsi = -0.5 * laplacian5(self._psi_bc, g.dx, g.dy) + self.V * self._psi_bc
        dpsi_dt = -1j * Hpsi
        self.bc.apply_rhs(dpsi_dt)
        return dpsi_dt.reshape(-1)

    def _maybe_renormalize(self):
        self._steps += 1
        if self._steps % self.renormalize_every == 0:
            g = self.grid
            psi = self.state.reshape(g.nx, g.ny)
            norm = (np.abs(psi)**2).sum() * g.dx * g.dy
            if norm > 0:
                psi /= np.sqrt(norm)

    # def init_draw(self, ax: plt.Axes) -> Sequence:
    #     # ensure a 3D axes and cache it
    #     if ax.name != "3d":
    #         fig = ax.figure
    #         ax.remove()
    #         ax = fig.add_subplot(111, projection="3d")

    #     self._ax3d = ax  # <— store the axes
    #     Z = np.abs(self.state.reshape(self.grid.nx, self.grid.ny))**2
    #     self._surf = ax.plot_surface(
    #         self.grid.X, self.grid.Y, Z,
    #         cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
    #     )
    #     ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("|psi|^2")
    #     # Fixed z-limit for consistent and faster redraws
    #     ax.set_zlim(0, self._zmax_fixed)
    #     return (self._surf,)

    # def update_draw(self, artists: Sequence) -> Sequence:
    #     ax = self._ax3d             # <— always use the cached axes
    #     (surf,) = artists
    #     try:
    #         surf.remove()
    #     except Exception:
    #         pass

    #     Z = np.abs(self.state.reshape(self.grid.nx, self.grid.ny))**2
    #     self._surf = ax.plot_surface(
    #         self.grid.X, self.grid.Y, Z,
    #         cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
    #     )
    #     # keep z-limits fixed to avoid expensive autoscaling
    #     ax.set_zlim(0, self._zmax_fixed)
    #     # self._maybe_renormalize()
    #     return (self._surf,)
    def init_draw(self, ax: plt.Axes):
        # 2D heatmap of |psi|^2 with barrier overlay for clarity
        Z = np.abs(self.state.reshape(self.grid.nx, self.grid.ny))**2
        im = ax.imshow(
            Z.T,
            origin="lower",
            extent=[0, self.grid.Lx, 0, self.grid.Ly],
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_xlabel("x"); ax.set_ylabel("y")

        # Overlay the double-slit barrier (V>0) as a semi-transparent mask
        try:
            mask = (self.V > 0.0).T  # match imshow orientation
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=float)
            overlay[mask] = [1.0, 0.0, 0.0, 0.35]  # red with alpha
            ax.imshow(
                overlay,
                origin="lower",
                extent=[0, self.grid.Lx, 0, self.grid.Ly],
                interpolation="nearest",
                zorder=3,
            )
        except Exception:
            pass

        self._im = im
        return (im,)

    def update_draw(self, artists):
        (im,) = artists
        Z = np.abs(self.state.reshape(self.grid.nx, self.grid.ny))**2
        im.set_data(Z.T)                 # update pixels only
        im.set_clim(0, max(1e-12, Z.max()))
        self._maybe_renormalize()
        return (im,)
