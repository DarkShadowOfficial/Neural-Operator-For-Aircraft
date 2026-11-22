"""OperatorService: fast prediction wrapper around a trained ensemble, bound to one FEM surface mesh."""
from __future__ import annotations

import numpy as np

from . import geometry
from .config import PNAMES, P_HI, P_LO, MACH_MAX, MACH_MIN, mach_number
from .mesh import aircraft_mesh
from .operator import DeepONet, Ensemble, drive_scale, initial_field, static_features


class OperatorService:
    def __init__(self, bundle: dict):
        self.meta = bundle["meta"]
        self.mesh_name = self.meta["mesh"]
        mesh = aircraft_mesh(self.mesh_name)
        self.xyz, self.nrm, self.tag = mesh.surf_xyz, mesh.surf_nrm, mesh.surf_tag
        self.n_surface = len(self.xyz)
        self.static = static_features(self.xyz, self.nrm, self.tag)
        self.ens = Ensemble([DeepONet.from_state(c, s) for c, s in zip(bundle["model_cfgs"], bundle["model_states"])])
        self.A = self.ens.modes("full", self.static)                   # trunk modes, computed once per mesh
        self.probes = []
        for name, p in geometry.PROBES:
            j = int(np.argmin(np.linalg.norm(self.xyz - np.array(p), axis=1)))
            self.probes.append(dict(name=name, node=j, target=list(p), distance=float(np.linalg.norm(self.xyz[j] - p))))

    def has_uncertainty(self) -> bool:
        return len(self.ens.models) > 1

    def predict_surface(self, cond: dict, times, idx=None):
        """(T_mean[nt, S'], T_std[nt, S']) in kelvin at the mesh surface nodes (or the subset `idx`)."""
        P = np.array([[float(cond[n]) for n in PNAMES]])
        times = np.atleast_1d(np.asarray(times, float))
        sl = slice(None) if idx is None else idx
        ys = np.stack([m.predict_y(P[0], times, A[sl]) for m, A in zip(self.ens.models, self.A)])
        T = initial_field(P, self.xyz[sl])[0][None] + drive_scale(P)[0] * ys
        return T.mean(0), T.std(0)

    def check(self, cond: dict) -> list[str]:
        """Human-readable warnings when a case lies outside what the operator was trained on."""
        w = []
        for n, lo, hi in zip(PNAMES, P_LO, P_HI):
            if not (lo <= cond[n] <= hi):
                w.append(f"{n} = {cond[n]:g} is outside the trained range [{lo:g}, {hi:g}] -> extrapolation")
        M = float(mach_number(cond["velocity"], cond["T_air"]))
        if not (MACH_MIN <= M <= MACH_MAX):
            w.append(f"Mach {M:.2f} outside [{MACH_MIN}, {MACH_MAX}]")
        return w
