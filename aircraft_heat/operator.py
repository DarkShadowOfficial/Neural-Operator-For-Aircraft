"""Neural operator (DeepONet, pure NumPy with hand-written back-propagation).

    T(c, x, t) = T_ic(x; c) + S(c) * sqrt(t / t_max) * sum_k  B_k(c, t) * A_k(x)

* ``c``   : flight/material/initial-temperature conditions (the operator's input "function" parameters);
* ``A(x)``: *trunk* net on the geometry of a surface point (position, normal, incidence, boundary-layer run
            length, component) -- continuous in x, works for any surface point;
* ``B``   : *branch* net on the conditions and time;
* ``S(c)``: the driving temperature scale |T_recovery - T_init| (so the net learns O(1) quantities);
* ``sqrt(t)`` makes the initial condition exact at t = 0 and reproduces the early-time sqrt(t) heating law.

Because time sits on the branch side, one prediction costs ``n_times`` tiny branch evaluations, one trunk
evaluation per surface point and one matrix product -- no per-(point, time) network evaluation.
NumPy (not PyTorch) keeps the repository dependency-light; the maths ports line-by-line.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import aero, geometry
from .config import (CP_AIR, FLOW_DIR, GAMMA, GEOM_HI, GEOM_LO, LENGTH, MAX_TIME, R_AIR, Preset, isa_pressure,
                     mach_number, to_unit)
from .mesh import AIRCRAFT

F32 = np.float32
_CEN, _HALF = (GEOM_LO + GEOM_HI) / 2, (GEOM_HI - GEOM_LO) / 2
N_STATIC = 14
N_BRANCH = 12


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def static_features(xyz: np.ndarray, nrm: np.ndarray, tag: np.ndarray | None = None) -> np.ndarray:
    """Geometry-only descriptors of surface points (computed once per point set, cached by the caller)."""
    xyz, nrm = np.asarray(xyz, float), np.asarray(nrm, float)
    tag = geometry.tag_points(xyz) if tag is None else tag
    s = aero.run_length(AIRCRAFT, xyz, nrm, tag)
    s = np.maximum(s, np.maximum(2.0 * aero.R_EFF[tag], 0.05))
    sin_inc = -(nrm @ FLOW_DIR)
    onehot = np.eye(len(geometry.TAGS))[tag]
    return np.concatenate([(xyz - _CEN) / _HALF, nrm, sin_inc[:, None], np.log10(s)[:, None] / 1.5, onehot], 1).astype(F32)


def drive_scale(P: np.ndarray) -> np.ndarray:
    """S(c) in kelvin: |T_stagnation - T_init| (the largest possible skin temperature change), floored at 20 K."""
    P = np.atleast_2d(P)
    return np.maximum(np.abs(P[:, 1] + P[:, 0] ** 2 / (2.0 * CP_AIR) - P[:, 3]), 20.0)


def initial_field(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """T_ic(x) for every condition row: (n, S)."""
    P = np.atleast_2d(P)
    return P[:, 3:4] + P[:, 4:5] * (np.asarray(xyz)[None, :, 0] / LENGTH - 0.5)


_AERO: dict = {}


def aero_summary(P: np.ndarray, mesh_name: str = "coarse") -> np.ndarray:
    """(n, 2): area-mean heat-transfer coefficient and recovery temperature from the closed-form aero model.
    Cheap (<1 ms per condition) functions of the inputs that carry the dominant aerodynamic-heating physics."""
    from . import fem
    if mesh_name not in _AERO:
        fg = fem.get_solver(mesh_name).faces
        sel = np.arange(0, len(fg.area), max(1, len(fg.area) // 600))          # subsample faces for speed
        _AERO[mesh_name] = (fg, sel)
    fg, sel = _AERO[mesh_name]
    sub = aero.FaceGeometry(fg.xyz[sel], fg.nrm[sel], fg.area[sel], fg.tag[sel], fg.s_run[sel], fg.sin_inc[sel],
                            fg.base[sel], fg.r_eff[sel])
    out = []
    for p in np.atleast_2d(P):
        T_w = aero.initial_temperature(sub.xyz, p[3], p[4])
        h, taw, _ = aero.boundary_conditions(sub, p[0], p[1], p[2], T_w)
        out.append([np.average(h, weights=sub.area), np.average(taw, weights=sub.area)])
    return np.array(out)


def branch_features(P: np.ndarray, t: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """(n, 8) physical conditions and (n,) times -> (n, N_BRANCH) network inputs (dimensionless groups included)."""
    P = np.atleast_2d(np.asarray(P, float))
    t = np.broadcast_to(np.asarray(t, float), (len(P),))
    M = mach_number(P[:, 0], P[:, 1])
    t0 = P[:, 1] + P[:, 0] ** 2 / (2.0 * CP_AIR)
    k, rho, cp = P[:, 5], P[:, 6], P[:, 7]
    eff = np.sqrt(k * rho * cp)
    alpha = k / (rho * cp)
    tn = t / MAX_TIME
    if extra is None:
        extra = aero_summary(P)
    h_m, taw_m = extra[:, 0], extra[:, 1]
    # Only the physically relevant groups are fed to the network (conduction sees k and rho*cp; the boundary
    # condition sees h and T_aw), not the raw velocity / altitude / material numbers.
    f = np.column_stack([
        M / 4.5,
        (t0 - P[:, 3]) / 500.0,
        (taw_m - P[:, 3]) / 500.0,
        np.log10(h_m / 300.0),
        np.log10(eff) - 3.8,
        (np.log10(alpha) + 5.2) / 1.5,
        P[:, 4] / 100.0,
        tn, np.sqrt(tn), np.log10(tn + 0.005) / 2.0 + 1.0,
        np.log10(np.sqrt(alpha * np.maximum(t, 0.0)) + 1e-4) + 1.5,
        np.log10(h_m * np.sqrt(np.maximum(t, 0.0)) / eff + 1e-3) / 2.0 + 1.0,
    ])
    return f.astype(F32)


# --------------------------------------------------------------------------- #
# tiny MLP with manual gradients
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])).astype(F32))
            self.b.append(np.zeros(sizes[i + 1], F32))

    def forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            acts.append(x)
        return x, acts

    def backward(self, acts, g):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return gW, gb

    def params(self):
        return self.W + self.b


class DeepONet:
    def __init__(self, width=80, latent=40, n_fourier=16, sigma=2.5, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = dict(width=width, latent=latent, n_fourier=n_fourier, sigma=sigma, seed=seed)
        self.Bf = (rng.standard_normal((3, n_fourier)) * sigma).astype(F32)
        self.branch = MLP([N_BRANCH, 2 * width, 2 * width, latent], rng)
        self.trunk = MLP([N_STATIC + 2 * n_fourier, width, width, latent], rng)

    # trunk input incl. random Fourier features of the position
    def trunk_input(self, static):
        proj = 2.0 * np.pi * static[:, :3] @ self.Bf
        return np.concatenate([static, np.sin(proj), np.cos(proj)], 1).astype(F32)

    def params(self):
        return self.branch.params() + self.trunk.params()

    def state(self) -> dict:
        d = {"Bf": self.Bf}
        for name, net in (("branch", self.branch), ("trunk", self.trunk)):
            for i, (W, b) in enumerate(zip(net.W, net.b)):
                d[f"{name}_W{i}"], d[f"{name}_b{i}"] = W, b
        return d

    @classmethod
    def from_state(cls, cfg: dict, st: dict):
        m = cls(**cfg)
        m.Bf = st["Bf"].astype(F32)
        for name, net in (("branch", m.branch), ("trunk", m.trunk)):
            for i in range(len(net.W)):
                net.W[i], net.b[i] = st[f"{name}_W{i}"].astype(F32), st[f"{name}_b{i}"].astype(F32)
        return m

    def modes(self, static_in):
        return self.trunk.forward(static_in)[0]

    def predict_y(self, P, t, A):
        """Normalised response y(c, t, x) for one condition row P (8,) at times t (nt,) given trunk modes A (S, latent)."""
        t = np.atleast_1d(np.asarray(t, float))
        P1 = np.atleast_2d(P)
        feats = branch_features(np.repeat(P1, len(t), 0), t, np.repeat(aero_summary(P1), len(t), 0))
        B = self.branch.forward(feats)[0]
        return np.sqrt(t / MAX_TIME)[:, None].astype(F32) * (B @ A.T)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.wd = wd
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads, lr):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if self.wd and p.ndim == 2:
                p *= (1.0 - lr * self.wd)
            p -= (lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(F32)


@dataclass
class FieldSet:
    """Normalised FEM fields for a set of conditions."""
    P: np.ndarray            # (n, 8) physical
    times: np.ndarray        # (nt,)
    Y: np.ndarray            # (n, nt, S) (T - T_ic) / S(c)
    T_ic: np.ndarray         # (n, S)
    scale: np.ndarray        # (n,)
    extra: np.ndarray        # (n, 2) aero summary per condition


def make_fieldset(P, times, T, xyz) -> FieldSet:
    P = np.asarray(P, float)
    T_ic = initial_field(P, xyz)
    sc = drive_scale(P)
    Y = ((T - T_ic[:, None, :]) / sc[:, None, None]).astype(F32)
    return FieldSet(P, np.asarray(times, float), Y, T_ic, sc, aero_summary(P))


def _eval_loss(model, A, fs: FieldSet, w, max_pairs=4000, rng=None):
    n, nt, S = fs.Y.shape
    pairs = [(i, j) for i in range(n) for j in range(1, nt)]
    if len(pairs) > max_pairs:
        rng = rng or np.random.default_rng(0)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    ci = np.array([p[0] for p in pairs])
    ti = np.array([p[1] for p in pairs])
    tot, cnt = 0.0, 0
    for a in range(0, len(pairs), 512):
        sl = slice(a, a + 512)
        feats = branch_features(fs.P[ci[sl]], fs.times[ti[sl]], fs.extra[ci[sl]])
        B = model.branch.forward(feats)[0]
        pred = np.sqrt(fs.times[ti[sl]] / MAX_TIME)[:, None].astype(F32) * (B @ A.T)
        tot += float((w * (pred - fs.Y[ci[sl], ti[sl]]) ** 2).sum())
        cnt += pred.size
    return np.sqrt(tot / cnt)


def loss_and_grads(model: DeepONet, feats, st_in, tf, Y, w):
    """Weighted MSE of  tf * (B A^T)  against Y, and gradients w.r.t. all parameters (order of model.params())."""
    B, bacts = model.branch.forward(feats)
    A, tacts = model.trunk.forward(st_in)
    diff = tf * (B @ A.T) - Y
    loss = float((w * diff ** 2).mean())
    g = (2.0 / diff.size) * w * diff * tf               # d loss / d (B A^T)
    gWb, gbb = model.branch.backward(bacts, g @ A)
    gWt, gbt = model.trunk.backward(tacts, g.T @ B)
    return loss, gWb + gbb + gWt + gbt


def train_model(model: DeepONet, train: FieldSet, val: FieldSet | None, static, w_nodes, cfg: Preset,
                epochs: int | None = None, seed: int = 0, progress=None, label="model", log_every=500,
                cond_probs: np.ndarray | None = None, warm_start: bool = False):
    """Mini-batch Adam; every step uses `nb` random (condition, time) pairs and ALL surface points.

    cond_probs : optional sampling probability per training condition (used to oversample hard cases);
    warm_start : continue from the model's current weights; the starting weights count as the first
                 candidate for "best validation state", so fine-tuning can never make validation worse."""
    rng = np.random.default_rng(seed)
    epochs = epochs or cfg.epochs
    nb = max(16, cfg.batch // 32)
    n, nt, S = train.Y.shape
    st_in = model.trunk_input(static)
    w = (w_nodes / w_nodes.mean()).astype(F32)[None, :]
    opt = Adam(model.params(), cfg.lr, wd=cfg.wd)
    best, best_state, hist = np.inf, None, []
    if warm_start and val is not None:
        best = _eval_loss(model, model.modes(st_in), val, w, rng=np.random.default_rng(1))
        best_state = {k: v.copy() for k, v in model.state().items()}
    t0 = time.time()
    for ep in range(epochs):
        lr = 1e-5 + 0.5 * (cfg.lr - 1e-5) * (1 + np.cos(np.pi * ep / epochs))
        ci = rng.integers(n, size=nb) if cond_probs is None else rng.choice(n, size=nb, p=cond_probs)
        ti = rng.integers(1, nt, size=nb)
        feats = branch_features(train.P[ci], train.times[ti], train.extra[ci])
        tf = np.sqrt(train.times[ti] / MAX_TIME)[:, None].astype(F32)
        loss, grads = loss_and_grads(model, feats, st_in, tf, train.Y[ci, ti], w)
        gn = np.sqrt(sum(float((x * x).sum()) for x in grads))
        if gn > 5.0:
            grads = [x * (5.0 / gn) for x in grads]
        opt.step(grads, lr)
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            vl = _eval_loss(model, model.modes(st_in), val, w, rng=np.random.default_rng(1)) if val is not None else loss ** 0.5
            hist.append((ep + 1, float(np.sqrt(loss)), float(vl)))
            if vl < best:
                best = vl
                best_state = {k: v.copy() for k, v in model.state().items()}
            if progress:
                progress.update((ep + 1) / epochs, f"{label}: step {ep + 1}/{epochs}  train rmse {np.sqrt(loss):.4f}  val rmse {vl:.4f}")
    if best_state is not None:
        m2 = DeepONet.from_state(model.cfg, best_state)
        model.branch, model.trunk, model.Bf = m2.branch, m2.trunk, m2.Bf
    return model, hist, time.time() - t0


class Ensemble:
    """Mean and spread of several independently trained DeepONets (the spread is the uncertainty estimate)."""

    def __init__(self, models: list[DeepONet]):
        self.models = models
        self._modes_cache: dict = {}

    def modes(self, key, static):
        if key not in self._modes_cache:
            self._modes_cache[key] = [m.modes(m.trunk_input(static)) for m in self.models]
        return self._modes_cache[key]

    def predict(self, P, times, key, static, xyz):
        """-> (T_mean[nt, S], T_std[nt, S]) in kelvin for one condition row P."""
        P = np.asarray(P, float).reshape(1, -1)
        ys = np.stack([m.predict_y(P[0], times, A) for m, A in zip(self.models, self.modes(key, static))])
        sc = drive_scale(P)[0]
        T_ic = initial_field(P, xyz)[0]
        T = T_ic[None] + sc * ys
        return T.mean(0), T.std(0)
