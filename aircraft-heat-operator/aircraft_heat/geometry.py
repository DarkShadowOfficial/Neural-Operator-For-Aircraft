"""Parametric supersonic aircraft as an implicit surface (signed-distance-like function).

Components (all inside the 20 x 13 x 5.2 m CAD box used for the SimScale runs):
fuselage with blunt elliptical nose, swept delta-type wing, swept vertical fin, ventral engine pod.
Coordinates: x along the fuselage (nose at 0), y span, z up.  Air flows in +x.
"""
from __future__ import annotations

import numpy as np

TAGS = ["nose", "fuselage", "aft", "wing", "fin", "pod"]

# fuselage
_LN = 5.5                           # elliptical nose length
_A, _B, _ZC = 1.15, 0.95, 0.70      # half width, half height, centre line height
# main wing
_WX0, _WSWEEP, _WTE, _WTIP, _WZ, _WTC, _WTMIN = 7.0, 1.4281, 18.5, 6.5, 0.50, 0.045, 0.03
# vertical fin
_FX0, _FSWEEP, _FTE, _FZ0, _FZ1, _FTC, _FTMIN = 13.2, 0.85, 19.8, 1.0, 4.2, 0.04, 0.03
# engine pod
_PX0, _PX1, _PR, _PZ = 9.5, 19.3, 0.75, -0.25


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _fuselage_shape(x):
    s = np.clip(x / _LN, 0.0, 1.0)
    nose = np.sqrt(np.clip(1.0 - (1.0 - s) ** 2, 0.0, 1.0))
    boat = 1.0 - 0.55 * _smoothstep((x - 13.0) / 7.0)
    return np.where(x < _LN, nose, boat)


def _f_fuselage(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2] - _ZC
    sh = np.maximum(_fuselage_shape(x), 0.03)
    a, b = _A * sh, _B * sh
    q = np.sqrt((y / a) ** 2 + (z / b) ** 2)
    f = (q - 1.0) * np.minimum(a, b)
    return np.maximum(f, np.maximum(-x, x - 20.0))


def _f_wing(p):
    x, y, z = p[..., 0], np.abs(p[..., 1]), p[..., 2] - _WZ
    xle = _WX0 + _WSWEEP * y
    c = np.maximum(_WTE - xle, 1e-3)
    xi = np.clip((x - xle) / c, 0.0, 1.0)
    th = _WTMIN + 0.5 * _WTC * c * 4.0 * xi * (1.0 - xi)
    cosl = 1.0 / np.sqrt(1.0 + _WSWEEP ** 2)
    return np.maximum.reduce([np.abs(z) - th, (xle - x) * cosl, x - _WTE, y - _WTIP])


def _f_fin(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    xle = _FX0 + _FSWEEP * (np.maximum(z, _FZ0) - 1.2)
    c = np.maximum(_FTE - xle, 1e-3)
    xi = np.clip((x - xle) / c, 0.0, 1.0)
    th = _FTMIN + 0.5 * _FTC * c * 4.0 * xi * (1.0 - xi)
    cosl = 1.0 / np.sqrt(1.0 + _FSWEEP ** 2)
    return np.maximum.reduce([np.abs(y) - th, (xle - x) * cosl, x - _FTE, z - _FZ1, _FZ0 - z])


def _f_pod(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2] - _PZ
    dx = np.maximum(np.maximum(_PX0 - x, x - _PX1), 0.0)
    f = np.sqrt(dx ** 2 + y ** 2 + z ** 2) - _PR
    return np.maximum(f, x - 20.0)


_COMPONENTS = (_f_fuselage, _f_wing, _f_fin, _f_pod)


def implicit(p):
    """Union of all components (negative inside). Not an exact distance; see `sdf`."""
    p = np.asarray(p, float)
    return np.minimum.reduce([f(p) for f in _COMPONENTS])


def gradient(p, eps=2e-4):
    p = np.asarray(p, float)
    g = np.empty_like(p)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        g[..., i] = (implicit(p + d) - implicit(p - d)) / (2 * eps)
    return g


def sdf(p, eps=2e-4):
    """First-order signed distance f / |grad f| (accurate near the surface)."""
    f = implicit(p)
    gn = np.linalg.norm(gradient(p, eps), axis=-1)
    return f / np.maximum(gn, 1e-6)


def normals(p, eps=2e-4):
    g = gradient(p, eps)
    return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)


def project_to_surface(p, iters=6):
    p = np.asarray(p, float).copy()
    for _ in range(iters):
        f = implicit(p)
        g = gradient(p)
        p -= (f / np.maximum((g ** 2).sum(-1), 1e-12))[..., None] * g
    return p


def tag_points(p):
    """Component index into TAGS for each point (the component whose surface is nearest)."""
    p = np.asarray(p, float)
    fs = np.stack([f(p) for f in _COMPONENTS])          # fuselage, wing, fin, pod
    k = fs.argmin(0)
    x = p[..., 0]
    out = np.where(k == 0, np.where(x < 4.5, 0, np.where(x > 14.0, 2, 1)), k + 2)
    return out.astype(np.int8)


def fuselage_arclength(x):
    """Arc length along the fuselage meridian from the nose (m), used for boundary-layer run length."""
    xs = np.linspace(0.0, 20.0, 4001)
    r = _fuselage_shape(xs) * 0.5 * (_A + _B)
    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(r) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return np.interp(x, xs, s)


# Named probe locations ("parts of the aircraft") used for the time-history plots.
PROBES = [
    ("Nose tip",              (0.00, 0.0, 0.70)),
    ("Fuselage top (mid)",    (9.00, 0.0, 0.70 + _B)),
    ("Fuselage belly (mid)",  (6.00, 0.0, 0.70 - _B)),
    ("Wing leading edge",     (_WX0 + _WSWEEP * 3.2, 3.2, _WZ)),
    ("Wing upper surface",    (_WX0 + _WSWEEP * 3.0 + 0.5 * (_WTE - _WX0 - _WSWEEP * 3.0), 3.0, _WZ + 0.1)),
    ("Wing tip",              (_WX0 + _WSWEEP * 6.4 + 0.6, 6.45, _WZ)),
    ("Wing trailing edge",    (_WTE, 3.0, _WZ)),
    ("Fin leading edge",      (_FX0 + _FSWEEP * (2.7 - 1.2), 0.0, 2.7)),
    ("Fin tip",               (_FX0 + _FSWEEP * 3.0 + 1.5, 0.0, 4.19)),
    ("Engine pod lip",        (_PX0 - _PR, 0.0, _PZ)),
    ("Tail cone",             (19.9, 0.0, 0.70)),
]
