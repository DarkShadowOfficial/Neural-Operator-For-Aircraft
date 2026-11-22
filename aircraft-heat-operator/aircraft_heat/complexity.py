"""Measured scaling of FEM vs operator cost (the "time complexity" comparison)."""
from __future__ import annotations

import time

import numpy as np

from . import fem
from .analysis import timed
from .config import MAX_TIME, P_DEFAULT
from .mesh import aircraft_mesh


def fem_scaling(meshes=("coarse", "medium"), progress=None) -> list[dict]:
    """FEM wall time for one scenario on meshes of increasing size (mesh build time reported separately)."""
    rows = []
    c = dict(P_DEFAULT)
    c["velocity"], c["T_air"] = 1000.0, 400.0
    for i, name in enumerate(meshes):
        t0 = time.perf_counter()
        mesh = aircraft_mesh(name)
        build = time.perf_counter() - t0
        solver = fem.get_solver(name)
        if progress:
            progress.update(i / len(meshes), f"benchmark: FEM on the {name} mesh ({mesh.n_nodes} nodes)")
        r = solver.solve(c, np.linspace(0, MAX_TIME, 61))
        rows.append(dict(mesh=name, n_nodes=int(mesh.n_nodes), n_tets=int(len(mesh.tets)),
                         n_surface=int(len(mesh.surf_nodes)), fem_s=float(r.info["wall_s"]),
                         factor_s=float(r.info["setup_s"]), steps=int(r.info["steps"]), mesh_build_s=float(build)))
    return rows


def operator_scaling(service) -> dict:
    """Operator wall time versus number of output times and number of surface points."""
    c = dict(P_DEFAULT)
    c["velocity"], c["T_air"] = 1000.0, 400.0
    out = dict(vs_times=[], vs_points=[])
    for nt in (5, 21, 61, 121, 241):
        _, s = timed(lambda: service.predict_surface(c, np.linspace(0, MAX_TIME, nt)), 5)
        out["vs_times"].append(dict(n=nt, s=s))
    for frac in (0.1, 0.25, 0.5, 1.0):
        idx = np.arange(0, service.n_surface, max(1, int(round(1 / frac))))
        _, s = timed(lambda: service.predict_surface(c, np.linspace(0, MAX_TIME, 61), idx), 5)
        out["vs_points"].append(dict(n=int(len(idx)), s=s))
    return out
