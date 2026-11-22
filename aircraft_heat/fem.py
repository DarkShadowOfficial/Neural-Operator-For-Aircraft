"""Transient 3-D heat conduction FEM (linear tetrahedra) with aerodynamic-heating boundary conditions.

    rho c dT/dt = div(k grad T)  in the solid,      -k dT/dn = h (T - T_aw)  on the wetted surface

* P1 tetrahedra, lumped (diagonal) capacity matrix, lumped Robin boundary terms;
* TR-BDF2 time integration (gamma = 2 - sqrt 2): second-order, L-stable (no ringing after the
  suddenly applied heat flux) and, because the two stages share one matrix, only ONE sparse LU
  factorisation per run;
* h and T_aw come from `aero.boundary_conditions` and are constant in time, so the system is
  linear time-invariant.  This is the "SimScale-like" reference that replaces a CHT run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from . import aero
from .config import MACH_MAX, MACH_MIN, MAX_TIME, mach_number
from .mesh import Mesh, aircraft_mesh

_GAMMA = 2.0 - np.sqrt(2.0)


def check_conditions(c: dict, strict: bool = True):
    M = float(mach_number(c["velocity"], c["T_air"]))
    if strict and not (MACH_MIN <= M <= MACH_MAX):
        raise ValueError(f"Mach {M:.2f} is outside the validity range [{MACH_MIN}, {MACH_MAX}] of the "
                         f"aerodynamic heating correlations (change speed or air temperature)")
    for k in ("k", "rho", "cp"):
        if c[k] <= 0:
            raise ValueError(f"{k} must be positive")
    return M


@dataclass
class FEMResult:
    times: np.ndarray            # (nt,)
    T_surf: np.ndarray           # (nt, S) K at the mesh surface nodes
    T_full: np.ndarray | None    # (nt, N) K at all nodes (only if requested)
    info: dict


class HeatFEM:
    """Reusable solver bound to one mesh (the mesh is built once; each run assembles + factorises)."""

    def __init__(self, mesh: Mesh | str = "coarse", dt: float = 0.5, h_scale: float = 1.0):
        self.mesh = aircraft_mesh(mesh) if isinstance(mesh, str) else mesh
        self.dt, self.h_scale = dt, h_scale
        self.faces = aero.face_geometry(self.mesh)
        self._nodes_of_face = self.mesh.faces
        self.T_wall_face_x = self.mesh.nodes[self.mesh.faces].mean(1)[:, 0]

    # ----------------------------------------------------------------------------------
    def boundary(self, c: dict):
        T_wall = aero.initial_temperature(self.faces.xyz, c["T_init"], c["dT_axial"])
        return aero.boundary_conditions(self.faces, c["velocity"], c["T_air"], c["altitude"], T_wall, self.h_scale)

    def solve(self, c: dict, times, return_full: bool = False, strict: bool = True) -> FEMResult:
        t_start = time.perf_counter()
        check_conditions(c, strict)
        mesh = self.mesh
        times = np.atleast_1d(np.asarray(times, float))
        t_end = float(times.max())
        n_steps = max(1, int(np.ceil(t_end / self.dt - 1e-9)))
        h = t_end / n_steps if t_end > 0 else self.dt

        hf, taw, diag = self.boundary(c)
        N = mesh.n_nodes
        T0 = aero.initial_temperature(mesh.nodes, c["T_init"], c["dT_axial"])
        snaps, t_setup = integrate(mesh, c["k"], c["rho"], c["cp"], hf, taw, T0, n_steps, h, t_start)
        snaps = np.array(snaps)
        grid = np.linspace(0.0, t_end, n_steps + 1) if t_end > 0 else np.array([0.0])
        out = _interp_time(grid, snaps, times)
        wall = time.perf_counter() - t_start
        info = dict(wall_s=wall, setup_s=t_setup, steps=n_steps, dt=h, n_nodes=N, n_tets=len(mesh.tets),
                    h_scale=self.h_scale, aero=diag)
        return FEMResult(times, out[:, mesh.surf_nodes], out if return_full else None, info)


def integrate(mesh: Mesh, k, rho, cp, h_face, taw_face, T0, n_steps, h, t_start=None):
    """TR-BDF2 march of  rho*cp*M dT/dt + (k K + H) T = F  for n_steps steps of size h.
    Returns (snapshots[n_steps+1, N], setup_seconds)."""
    t_start = time.perf_counter() if t_start is None else t_start
    N = mesh.n_nodes
    w = h_face * mesh.face_area / 3.0
    Hd = np.zeros(N)
    F = np.zeros(N)
    for i in range(3):
        Hd += np.bincount(mesh.faces[:, i], weights=w, minlength=N)
        F += np.bincount(mesh.faces[:, i], weights=w * taw_face, minlength=N)
    M = rho * cp * mesh.m0
    A = k * mesh.K0 + sp.diags(Hd)
    coef = 2.0 / (_GAMMA * h)
    lu = splu((sp.diags(coef * M) + A).tocsc())
    t_setup = time.perf_counter() - t_start
    T = np.asarray(T0, float).copy()
    snaps = [T.copy()]
    g2 = _GAMMA * (2.0 - _GAMMA)
    for _ in range(n_steps):
        Ts = lu.solve(coef * M * T - A @ T + 2.0 * F)
        T = lu.solve(coef * M * (Ts / g2 - (1.0 - _GAMMA) ** 2 * T / g2) + F)
        snaps.append(T.copy())
    return np.array(snaps), t_setup


def _interp_time(grid, snaps, times):
    if len(grid) == 1:
        return np.repeat(snaps, len(times), axis=0)
    j = np.clip(np.searchsorted(grid, times, side="right") - 1, 0, len(grid) - 2)
    f = ((times - grid[j]) / (grid[j + 1] - grid[j]))[:, None]
    return snaps[j] * (1 - f) + snaps[j + 1] * f


_SOLVERS: dict = {}


def get_solver(mesh_name: str = "coarse", dt: float = 0.5) -> HeatFEM:
    key = (mesh_name, dt)
    if key not in _SOLVERS:
        _SOLVERS[key] = HeatFEM(mesh_name, dt)
    return _SOLVERS[key]


def solve_fem(c: dict, times, mesh_name: str = "coarse", dt: float = 0.5, **kw) -> FEMResult:
    return get_solver(mesh_name, dt).solve(c, times, **kw)
