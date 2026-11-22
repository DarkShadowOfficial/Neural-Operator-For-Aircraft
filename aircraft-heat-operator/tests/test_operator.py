"""Unit tests of the NumPy neural operator: gradients, exact initial condition, save/load, reproducibility."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aircraft_heat import config                                          # noqa: E402
from aircraft_heat.operator import (DeepONet, Ensemble, branch_features, drive_scale, initial_field,  # noqa: E402
                                    loss_and_grads, static_features)
from aircraft_heat.mesh import aircraft_mesh                              # noqa: E402


def _tiny():
    rng = np.random.default_rng(0)
    m = DeepONet(width=7, latent=5, n_fourier=3, seed=1)
    for net in (m.branch, m.trunk):                                       # float64 for finite differences
        net.W = [w.astype(np.float64) for w in net.W]
        net.b = [(b + 0.1 * rng.standard_normal(b.shape)).astype(np.float64) for b in net.b]
    S = 9
    static = rng.standard_normal((S, 14))
    P = config.sample_conditions(4, rng)
    feats = branch_features(P, np.array([10.0, 30.0, 60.0, 120.0]), np.column_stack([np.full(4, 300.0), np.full(4, 600.0)])).astype(np.float64)
    tf = np.sqrt(np.array([10.0, 30.0, 60.0, 120.0]) / config.MAX_TIME)[:, None]
    Y = rng.standard_normal((4, S))
    w = rng.random((1, S)) + 0.5
    return m, m.trunk_input(static).astype(np.float64), feats, tf, Y, w


def test_gradients_match_finite_differences():
    m, st, feats, tf, Y, w = _tiny()
    loss, grads = loss_and_grads(m, feats, st, tf, Y, w)
    rng = np.random.default_rng(3)
    worst = 0.0
    for p, g in zip(m.params(), grads):
        for _ in range(4):
            idx = tuple(rng.integers(s) for s in p.shape)
            old = p[idx]
            h = 1e-6
            p[idx] = old + h
            lp = loss_and_grads(m, feats, st, tf, Y, w)[0]
            p[idx] = old - h
            lm = loss_and_grads(m, feats, st, tf, Y, w)[0]
            p[idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx]) + 1e-12))
    assert worst < 1e-5, worst


def test_initial_condition_is_exact_and_save_load_roundtrip():
    mesh = aircraft_mesh("coarse")
    static = static_features(mesh.surf_xyz[:200], mesh.surf_nrm[:200], mesh.surf_tag[:200])
    m = DeepONet(width=16, latent=8, n_fourier=4, seed=2)
    ens = Ensemble([m])
    P = np.array(list(config.P_DEFAULT.values()))
    T, std = ens.predict(P, np.array([0.0, 50.0]), "t", static, mesh.surf_xyz[:200])
    ic = initial_field(P[None], mesh.surf_xyz[:200])[0]
    assert np.allclose(T[0], ic)                                            # t = 0 reproduces the initial field exactly
    m2 = DeepONet.from_state(m.cfg, m.state())
    T2, _ = Ensemble([m2]).predict(P, np.array([0.0, 50.0]), "t2", static, mesh.surf_xyz[:200])
    assert np.allclose(T, T2, atol=1e-4)


def test_unit_scaling_roundtrip():
    rng = np.random.default_rng(0)
    v = config.sample_conditions(20, rng)
    assert np.allclose(config.from_unit(config.to_unit(v)), v, rtol=1e-9, atol=1e-9)
    assert (config.to_unit(v) >= -1e-9).all() and (config.to_unit(v) <= 1 + 1e-9).all()
    assert (drive_scale(v) >= 20).all()


if __name__ == "__main__":
    test_gradients_match_finite_differences()
    test_initial_condition_is_exact_and_save_load_roundtrip()
    test_unit_scaling_roundtrip()
    print("operator tests passed")
