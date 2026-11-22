"""Central configuration: parameter ranges, materials, constants, training presets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Physical constants (ideal gas "air")
# --------------------------------------------------------------------------- #
GAMMA = 1.4
R_AIR = 287.05          # J/kg/K
CP_AIR = GAMMA * R_AIR / (GAMMA - 1.0)
PR = 0.71               # Prandtl number

# --------------------------------------------------------------------------- #
# Aircraft box (same CAD box as the SimScale notebook) and time horizon
# --------------------------------------------------------------------------- #
GEOM_LO = np.array([0.0, -6.5, -1.0])
GEOM_HI = np.array([20.0, 6.5, 4.2])
LENGTH = 20.0
MAX_TIME = 120.0        # s, simulated flight segment (the SimScale runs last 88-113 s)
FLOW_DIR = np.array([1.0, 0.0, 0.0])   # air moves +x relative to the aircraft; nose at x = 0

# --------------------------------------------------------------------------- #
# Operator inputs: name, label, unit, low, high, log-scale, default
# Ranges are the span the operator is trained on; the FEM itself accepts any value
# inside the validity limits of the aerodynamic correlations (MACH_MIN..MACH_MAX).
# --------------------------------------------------------------------------- #
PARAMS = [
    # name        label                     unit      lo      hi      log    default
    ("velocity", "Flight speed",             "m/s",    100.0, 1800.0, False, 1000.0),
    ("T_air",    "Air temperature",          "K",      220.0,  700.0, False,  400.0),
    ("altitude", "Altitude",                 "m",        0.0, 15000., False,    0.0),
    ("T_init",   "Initial aircraft temp.",   "K",      220.0,  700.0, False,  350.0),
    ("dT_axial", "Nose-to-tail initial dT",  "K",     -100.0,  100.0, False,    0.0),
    ("k",        "Conductivity k",           "W/m/K",    1.0,  250.0, True,     6.7),
    ("rho",      "Density",                  "kg/m3",  1500., 9000., False,  4430.0),
    ("cp",       "Heat capacity",            "J/kg/K",  400., 1100., False,   526.0),
]
PNAMES = [p[0] for p in PARAMS]
P_LO = np.array([p[3] for p in PARAMS])
P_HI = np.array([p[4] for p in PARAMS])
P_LOG = np.array([p[5] for p in PARAMS])
P_DEFAULT = {p[0]: p[6] for p in PARAMS}
N_PARAMS = len(PARAMS)
_LLO = np.log(np.where(P_LOG, P_LO, 1.0))      # log bounds (only used for log-scaled parameters)
_LHI = np.log(np.where(P_LOG, P_HI, 1.0))
_LDEN = np.where(P_LOG, _LHI - _LLO, 1.0)

MACH_MIN, MACH_MAX = 0.2, 4.5     # validity of the ideal-gas correlations used for the boundary layer

# Material presets (room-temperature handbook values, constant in the model).
MATERIALS = {
    "Ti-6Al-4V (titanium)":   dict(k=6.7,   rho=4430.0, cp=526.0),
    "Titanium alloy (notebook)": dict(k=7.0, rho=4500.0, cp=560.0),
    "Aluminium 2024-T3":      dict(k=121.0, rho=2780.0, cp=875.0),
    "Inconel 718":            dict(k=11.4,  rho=8190.0, cp=435.0),
    "Stainless steel 304":    dict(k=16.2,  rho=8000.0, cp=500.0),
    "Carbon/epoxy (through-thickness)": dict(k=2.0, rho=1600.0, cp=1000.0),
}


def to_unit(values) -> np.ndarray:
    """Physical parameter vector(s) (..., N_PARAMS) -> [0, 1]."""
    v = np.asarray(values, float)
    lv = np.log(np.clip(v, 1e-12, None))
    return np.where(P_LOG, (lv - _LLO) / _LDEN, (v - P_LO) / (P_HI - P_LO))


def from_unit(u) -> np.ndarray:
    u = np.asarray(u, float)
    return np.where(P_LOG, np.exp(_LLO + u * _LDEN), P_LO + u * (P_HI - P_LO))


def mach_number(velocity, T_air):
    return np.asarray(velocity, float) / np.sqrt(GAMMA * R_AIR * np.asarray(T_air, float))


def isa_pressure(altitude_m):
    """ISA static pressure (Pa) at altitude (m), valid to 20 km."""
    h = np.asarray(altitude_m, float)
    p_trop = 101325.0 * (1.0 - 2.25577e-5 * np.minimum(h, 11000.0)) ** 5.25588
    return np.where(h <= 11000.0, p_trop, 22632.1 * np.exp(-(h - 11000.0) / 6341.62))


def params_dict_to_vec(c: dict) -> np.ndarray:
    return np.array([float(c[n]) for n in PNAMES])


def sample_conditions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube style draw of n physical conditions inside the validity region."""
    out: list = []
    while len(out) < n:
        m = 2 * (n - len(out)) + 8
        u = np.empty((m, N_PARAMS))
        for j in range(N_PARAMS):
            u[:, j] = (rng.permutation(m) + rng.random(m)) / m
        v = from_unit(u)
        M = mach_number(v[:, 0], v[:, 1])
        ok = (M >= MACH_MIN) & (M <= MACH_MAX)
        out.extend(v[ok].tolist())
    return np.array(out[:n])


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
@dataclass
class Preset:
    name: str
    mesh: str = "coarse"           # "coarse" | "medium" | "fine"  (FEM mesh used for data + demo)
    n_train: int = 100             # FEM runs used for training
    n_val: int = 12                # FEM runs for early stopping / model selection
    n_test: int = 16               # held-out FEM runs, never touched in training
    n_times: int = 41              # output times per run used for training
    ensemble: int = 3
    epochs: int = 4000
    batch: int = 4096
    width: int = 96
    latent: int = 48
    lr: float = 2e-3
    n_fourier: int = 24
    seed: int = 0
    wd: float = 0.0


PRESETS = {
    "smoke": Preset("smoke", mesh="coarse", n_train=8, n_val=3, n_test=3, n_times=21,
                    ensemble=2, epochs=300, width=48, latent=24, n_fourier=8),
    "quick": Preset("quick", mesh="coarse", n_train=100, n_val=8, n_test=16, n_times=31,
                    ensemble=3, epochs=16000, width=128, latent=48, n_fourier=16),
    "full":  Preset("full", mesh="medium", n_train=400, n_val=24, n_test=32, n_times=41,
                    ensemble=5, epochs=30000, width=192, latent=64, n_fourier=24),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
