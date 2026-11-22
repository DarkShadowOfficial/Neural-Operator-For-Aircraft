"""Aerodynamic heating model: the stand-in for the CFD half of a SimScale conjugate-heat-transfer run.

For every wetted surface triangle it returns a heat-transfer coefficient ``h`` and an adiabatic-wall
(recovery) temperature ``T_aw`` so that the heat flux into the skin is ``q = h (T_aw - T_wall)``.

Engineering model (all standard textbook correlations, ideal-gas air):

* local edge state from modified-Newtonian pressure on windward faces (normal-shock total pressure for
  M > 1) and free-stream state on leeward faces, then isentropic expansion to the local edge Mach number;
* flat-plate boundary layer with Eckert's reference-temperature method, laminar -> turbulent blend
  on the reference Reynolds number, run length measured from the leading edge by ray marching;
* stagnation / leading-edge heating from Sutton-Graves, scaled by cos(incidence)^1.2 (swept-edge law);
* reduced heating on base (downstream-facing) surfaces.

``h_scale`` multiplies h so the model can be calibrated against SimScale results.
Validity: ideal gas, attached flow, M_inf between MACH_MIN and MACH_MAX; radiation is neglected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry
from .config import CP_AIR, FLOW_DIR, GAMMA, LENGTH, PR, R_AIR, isa_pressure
from .mesh import AIRCRAFT, Domain, Mesh

# effective nose / leading-edge radius per component (m): nose, fuselage, aft, wing, fin, pod
R_EFF = np.array([0.25, 6.0, 6.0, 0.03, 0.03, 0.75])
K_SG = 1.7415e-4          # Sutton-Graves constant, SI units
RE_TRANSITION = (5.7, 6.7)  # log10 of reference Reynolds number where transition starts / ends
BASE_FACTOR = 0.30


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def sutherland(T):
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


@dataclass
class FaceGeometry:
    """Condition-independent geometric data of the wetted faces (computed once per mesh)."""
    xyz: np.ndarray
    nrm: np.ndarray
    area: np.ndarray
    tag: np.ndarray
    s_run: np.ndarray       # boundary-layer run length (m)
    sin_inc: np.ndarray     # -n . flow_dir  (>0 windward)
    base: np.ndarray        # bool, downstream-facing
    r_eff: np.ndarray


def run_length(dom: Domain, xyz, nrm, tag, step=0.05, s_max=26.0):
    """Distance upstream (along -flow) from just inside each face until the solid is left."""
    q = xyz - 0.01 * nrm
    s = np.zeros(len(q))
    alive = np.ones(len(q), bool)
    u = -FLOW_DIR
    for i in range(1, int(s_max / step)):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        out = dom.implicit(q[idx] + i * step * u) > 0
        s[idx[out]] = i * step
        alive[idx[out]] = False
    s[alive] = s_max
    fus = tag <= 2
    s = np.where(fus, np.maximum(s, geometry.fuselage_arclength(xyz[:, 0])), s)
    return s


def face_geometry(mesh: Mesh, dom: Domain = AIRCRAFT) -> FaceGeometry:
    tag = geometry.tag_points(mesh.face_xyz)
    r_eff = R_EFF[tag]
    s = run_length(dom, mesh.face_xyz, mesh.face_nrm, tag)
    s = np.maximum(s, np.maximum(2.0 * r_eff, 0.05))
    sin_inc = -(mesh.face_nrm @ FLOW_DIR)
    return FaceGeometry(mesh.face_xyz, mesh.face_nrm, mesh.face_area, tag, s, sin_inc, sin_inc < -0.5, r_eff)


def freestream(velocity, T_air, altitude):
    U, T, p = float(velocity), float(T_air), float(isa_pressure(altitude))
    a = np.sqrt(GAMMA * R_AIR * T)
    M = U / a
    rho = p / (R_AIR * T)
    T0 = T + U ** 2 / (2 * CP_AIR)
    p0 = p * (1 + 0.2 * M ** 2) ** 3.5
    if M > 1.0:
        p02 = p * (((GAMMA + 1) * M ** 2 / 2) ** (GAMMA / (GAMMA - 1))
                   / ((2 * GAMMA * M ** 2 - (GAMMA - 1)) / (GAMMA + 1)) ** (1 / (GAMMA - 1)))
    else:
        p02 = p0
    return dict(U=U, T=T, p=p, a=a, M=M, rho=rho, T0=T0, p0=p0, p02=p02, q=0.5 * rho * U ** 2)


def boundary_conditions(fg: FaceGeometry, velocity, T_air, altitude, T_wall, h_scale=1.0):
    """Per-face (h [W/m2/K], T_aw [K], diagnostics) for the given flight condition."""
    fs = freestream(velocity, T_air, altitude)
    w = np.clip(fg.sin_inc, 0.0, 1.0)

    # local edge state
    cp_max = (fs["p02"] - fs["p"]) / fs["q"]
    p_e = fs["p"] + cp_max * fs["q"] * w ** 2
    p0_e = fs["p0"] + (fs["p02"] - fs["p0"]) * w ** 2
    ratio = np.maximum(p0_e / p_e, 1.0)
    M_e = np.sqrt(np.maximum(2.0 / (GAMMA - 1.0) * (ratio ** ((GAMMA - 1.0) / GAMMA) - 1.0), 1e-8))
    T_e = fs["T0"] / (1.0 + 0.5 * (GAMMA - 1.0) * M_e ** 2)
    u_e = M_e * np.sqrt(GAMMA * R_AIR * T_e)

    # flat plate, reference temperature (evaluated for laminar and turbulent, blended on Re*)
    def plate(c_m, c_T, k_st, expo, rec):
        T_ref = np.maximum(T_e * (1.0 + c_m * M_e ** 2 + c_T * (T_wall / T_e - 1.0)), 50.0)
        rho_ref = p_e / (R_AIR * T_ref)
        mu_ref = sutherland(T_ref)
        re = np.maximum(rho_ref * u_e * fg.s_run / mu_ref, 1.0)
        st = k_st * PR ** (-2.0 / 3.0) * re ** expo
        return st * rho_ref * u_e * CP_AIR, T_e * (1.0 + rec * 0.5 * (GAMMA - 1.0) * M_e ** 2), re

    h_l, taw_l, re_l = plate(0.032, 0.58, 0.332, -0.5, PR ** 0.5)
    h_t, taw_t, re_t = plate(0.035, 0.45, 0.0296, -0.2, PR ** (1.0 / 3.0))
    frac_t = _smoothstep((np.log10(re_l) - RE_TRANSITION[0]) / (RE_TRANSITION[1] - RE_TRANSITION[0]))
    h_fp = (1 - frac_t) * h_l + frac_t * h_t
    taw_fp = (1 - frac_t) * taw_l + frac_t * taw_t

    # stagnation / leading-edge heating
    h_st0 = (K_SG * np.sqrt(fs["rho"] / fg.r_eff) * fs["U"] ** 3 * CP_AIR
             / (0.5 * fs["U"] ** 2 + CP_AIR * max(fs["T"] - 300.0, 0.0)))
    h_st = h_st0 * w ** 1.2

    p4 = 4.0
    a4, b4 = h_fp ** p4, h_st ** p4
    h = (a4 + b4) ** (1.0 / p4)
    share = b4 / np.maximum(a4 + b4, 1e-300)
    taw = (1 - share) * taw_fp + share * fs["T0"]

    base = fg.base
    h = np.where(base, BASE_FACTOR * h_fp, h)
    taw = np.where(base, taw_fp, taw)
    h = h * h_scale
    diag = dict(M_inf=fs["M"], T0=fs["T0"], p_inf=fs["p"], rho_inf=fs["rho"], q_inf=fs["q"],
                h_min=float(h.min()), h_max=float(h.max()),
                h_mean=float(np.average(h, weights=fg.area)), taw_max=float(taw.max()),
                taw_mean=float(np.average(taw, weights=fg.area)), frac_turbulent=float(np.average(frac_t, weights=fg.area)))
    return h, taw, diag


def initial_temperature(xyz, T_init, dT_axial):
    """Initial skin temperature field: uniform plus a linear nose-to-tail difference."""
    return T_init + dT_axial * (np.asarray(xyz)[..., 0] / LENGTH - 0.5)
