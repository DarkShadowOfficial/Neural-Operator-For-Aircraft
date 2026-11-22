"""Verification of the FEM against exact solutions.  Run:  python -m pytest tests -q   (or python tests/test_fem_verification.py)

1. Semi-infinite solid with a convective surface (erfc solution)      -> checks conduction + Robin BC + time stepping
2. Small, very conductive block (lumped capacitance exp(-t/tau))      -> checks the capacity matrix and the surface integrals
3. Energy balance on the real aircraft mesh                            -> heat stored == heat delivered through the surface
4. Mesh convergence on the aircraft (coarse / medium)                  -> probe temperatures stop changing with refinement
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aircraft_heat import fem, geometry                                    # noqa: E402
from aircraft_heat.mesh import Domain, MeshSpec, build_mesh                # noqa: E402


def _box(lx, ly, lz):
    c = np.array([lx, ly, lz]) / 2

    def f(p):
        q = np.abs(np.asarray(p) - c) - c
        return np.max(q, axis=-1)
    return Domain(f, [0, 0, 0], [lx, ly, lz])


def run_semi_infinite_convection():
    lz = 0.5
    dom = _box(0.6, 0.6, lz)
    spec = MeshSpec(h_max=0.2, h_min=0.12, nl=14, d0=0.0015, d_cap=0.25, curv=0.5)
    mesh = build_mesh(dom, spec, face_filter=lambda x, n: x[:, 2] > lz - 1e-6)
    k, rho, cp, h, T0, Taw, t_end = 15.0, 7000.0, 500.0, 800.0, 300.0, 900.0, 60.0
    alpha = k / (rho * cp)
    assert 2 * np.sqrt(alpha * t_end) < 0.2 * lz               # slab is effectively semi-infinite
    n = 120
    snaps, _ = fem.integrate(mesh, k, rho, cp, np.full(len(mesh.faces), h), np.full(len(mesh.faces), Taw),
                             np.full(mesh.n_nodes, T0), n, t_end / n)

    def exact(depth, t):
        t = np.maximum(t, 1e-12)
        x = depth
        a = np.sqrt(alpha * t)
        return T0 + (Taw - T0) * (erfc(x / (2 * a)) - np.exp(h * x / k + h ** 2 * alpha * t / k ** 2) * erfc(x / (2 * a) + h * a / k))

    inner = (np.abs(mesh.nodes[:, 0] - 0.3) < 0.15) & (np.abs(mesh.nodes[:, 1] - 0.3) < 0.15)   # away from the adiabatic side walls
    top = np.where((mesh.nodes[:, 2] > lz - 1e-6) & inner)[0]
    errs = []
    for tt in (5.0, 15.0, 30.0, 60.0):
        T = snaps[int(round(tt / (t_end / n)))]
        err = np.abs(T[top] - exact(0.0, tt)) / (Taw - T0)
        errs.append(err.max())
    # sub-surface profile at the end time
    T = snaps[-1]
    dpt = lz - mesh.nodes[:, 2]
    sel = (dpt < 0.08) & inner & (dpt >= 0)
    prof = np.abs(T[sel] - exact(dpt[sel], t_end)) / (Taw - T0)
    print(f"[plate] surface error (max, % of driving dT) at t=5,15,30,60 s: {np.round(100 * np.array(errs), 3)}; "
          f"profile max {100 * prof.max():.3f} %  (nodes {mesh.n_nodes})")
    return max(errs), prof.max()


def run_lumped_capacitance():
    dom = _box(0.2, 0.2, 0.2)
    spec = MeshSpec(h_max=0.1, h_min=0.06, nl=6, d0=0.01, d_cap=0.1, curv=0.5)
    mesh = build_mesh(dom, spec)                           # all faces convective
    k, rho, cp, h, T0, Tinf, t_end = 400.0, 2700.0, 900.0, 50.0, 300.0, 600.0, 400.0
    V, A = mesh.m0.sum(), mesh.face_area.sum()             # discrete volume / area
    tau = rho * cp * V / (h * A)
    n = 200
    snaps, _ = fem.integrate(mesh, k, rho, cp, np.full(len(mesh.faces), h), np.full(len(mesh.faces), Tinf),
                             np.full(mesh.n_nodes, T0), n, t_end / n)
    t = np.linspace(0, t_end, n + 1)
    exact = Tinf + (T0 - Tinf) * np.exp(-t / tau)
    mean = (snaps * mesh.m0).sum(1) / V
    err = np.abs(mean - exact).max() / (Tinf - T0)
    spread = (snaps.max(1) - snaps.min(1)).max() / (Tinf - T0)      # Biot ~ 0.01 -> nearly uniform
    print(f"[lumped] volume-mean error {100 * err:.4f} % (discrete V, A used), spread {100 * spread:.2f} %")
    return err


def run_energy_balance_aircraft():
    S = fem.get_solver("coarse")
    mesh = S.mesh
    c = dict(velocity=1000., T_air=400., altitude=0., T_init=350., dT_axial=30., k=6.7, rho=4430., cp=526.)
    t_end, n = 120.0, 240
    hf, taw, _ = S.boundary(c)
    T0 = fem.aero.initial_temperature(mesh.nodes, c["T_init"], c["dT_axial"])
    snaps, _ = fem.integrate(mesh, c["k"], c["rho"], c["cp"], hf, taw, T0, n, t_end / n)
    M = c["rho"] * c["cp"] * mesh.m0
    stored = (M * (snaps[-1] - snaps[0])).sum()
    # heat delivered: integral over time of sum_f h A (Taw - T_face); trapezoid rule on the snapshots
    flux = []
    for T in snaps:
        Tf = T[mesh.faces].mean(1)
        flux.append((hf * mesh.face_area * (taw - Tf)).sum())
    delivered = np.trapezoid(flux, dx=t_end / n)
    rel = abs(stored - delivered) / abs(delivered)
    print(f"[energy] stored {stored / 1e6:.2f} MJ vs delivered {delivered / 1e6:.2f} MJ, relative difference {100 * rel:.3f} %")
    return rel


def run_mesh_convergence():
    c = dict(velocity=1000., T_air=400., altitude=0., T_init=350., dT_axial=0., k=6.7, rho=4430., cp=526.)
    times = np.array([30.0, 120.0])
    res = {}
    for name in ("coarse", "medium"):
        S = fem.get_solver(name)
        r = S.solve(c, times)
        res[name] = (S.mesh, r)
    rows = []
    for pname, p in geometry.PROBES:
        vals = []
        for name in ("coarse", "medium"):
            mesh, r = res[name]
            j = np.argmin(np.linalg.norm(mesh.surf_xyz - np.array(p), axis=1))
            vals.append(r.T_surf[:, j])
        rows.append((pname, vals[0], vals[1]))
    print("[convergence] probe temperatures at t = 30 s / 120 s  (coarse | medium)")
    worst = 0.0
    panels = {"Fuselage top (mid)", "Fuselage belly (mid)", "Wing upper surface", "Tail cone"}
    for pname, a, b in rows:
        if pname in panels:
            worst = max(worst, abs(a[1] - b[1]))                 # end-of-flight value on smooth panels
        print(f"   {pname:22s} {a[0]:7.1f} {a[1]:7.1f} | {b[0]:7.1f} {b[1]:7.1f}")
    for name in res:
        print(f"   {name}: nodes {res[name][0].n_nodes}, wall {res[name][1].info['wall_s']:.2f} s")
    return worst


def test_semi_infinite_convection():
    e, p = run_semi_infinite_convection()
    assert e < 0.02 and p < 0.01        # < 2 % of the driving temperature difference (worst at the first seconds)


def test_lumped_capacitance():
    assert run_lumped_capacitance() < 0.005


def test_energy_balance_aircraft():
    assert run_energy_balance_aircraft() < 1e-3


def test_mesh_convergence_on_panels():
    # smooth panels converge tightly; thin tips/edges are mesh-limited and only reported, not asserted
    assert run_mesh_convergence() < 20.0


if __name__ == "__main__":
    e1, p1 = run_semi_infinite_convection()
    e2 = run_lumped_capacitance()
    e3 = run_energy_balance_aircraft()
    w = run_mesh_convergence()
    print(f"\nsummary: plate {100 * e1:.3f}% | lumped {100 * e2:.3f}% | energy {100 * e3:.3f}% | worst smooth-panel probe diff coarse vs medium {w:.1f} K")
