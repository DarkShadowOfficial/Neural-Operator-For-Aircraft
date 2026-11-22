"""Layered ("skin") tetrahedral mesh generation for an implicit solid -- no external mesher needed.

Aerodynamic heating only penetrates a few centimetres of skin in the time spans of interest, so the
FEM domain is the *skin layer* of the solid, meshed the way boundary layers are meshed in CFD:

1. sample the surface and thin it to a Poisson-disk point set whose spacing follows local curvature and
   wall thickness (fine on leading edges, coarse on flat panels), triangulate it (via a Delaunay
   tetrahedralisation of surface + coarse interior points, keeping the faces on the true surface);
2. extrude every surface node inward along its normal in geometrically graded layers.  A column stops
   where the clearance to the nearest other surface drops (half the local wall thickness) or at
   ``d_cap``; at that depth the bottom of the column is an adiabatic (symmetry) plane;
3. split every prism into three tetrahedra with a global-index rule so neighbouring prisms conform.

Conduction inside the skin is fully 3-D (through-thickness and lateral); conduction deeper than the
column depth is neglected (``d_cap`` is chosen to exceed 2*sqrt(alpha*t) for the materials/time
span supported by the demo).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree

from . import geometry
from .config import GEOM_HI, GEOM_LO


class Domain:
    """An implicit solid: f(p) < 0 inside.  Derived helpers (gradient, distance, normal)."""

    def __init__(self, implicit: Callable, lo, hi):
        self.implicit = implicit
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def gradient(self, p, eps=2e-4):
        p = np.asarray(p, float)
        g = np.empty_like(p)
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            g[..., i] = (self.implicit(p + d) - self.implicit(p - d)) / (2 * eps)
        return g

    def sdf(self, p):
        g = np.linalg.norm(self.gradient(p), axis=-1)
        return self.implicit(p) / np.maximum(g, 1e-6)

    def normals(self, p):
        g = self.gradient(p)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)

    def project(self, p, iters=6):
        p = np.asarray(p, float).copy()
        for _ in range(iters):
            f = self.implicit(p)
            g = self.gradient(p)
            p -= (f / np.maximum((g ** 2).sum(-1), 1e-12))[..., None] * g
        return p


AIRCRAFT = Domain(geometry.implicit, GEOM_LO, GEOM_HI)


@dataclass(frozen=True)
class MeshSpec:
    h_max: float                  # coarsest surface spacing (m)
    h_min: float                  # finest surface spacing (m), at edges / small radii
    nl: int = 10                  # nodes per column (layers = nl - 1)
    d0: float = 0.003             # first layer thickness (m)
    d_cap: float = 0.45           # maximum column depth (m)
    grow: float = 0.5             # surface size grading rate (size change per metre)
    curv: float = 0.5             # surface spacing as a fraction of the local radius of curvature
    n_cand: int = 120_000         # surface candidates before thinning
    seed: int = 0


SPECS = {
    "coarse": MeshSpec(h_max=0.90, h_min=0.20, nl=9, curv=0.5),
    "medium": MeshSpec(h_max=0.60, h_min=0.12, nl=10, curv=0.35),
    "fine":   MeshSpec(h_max=0.40, h_min=0.08, nl=11, curv=0.25),
}


@dataclass
class Mesh:
    nodes: np.ndarray            # (N, 3)
    tets: np.ndarray             # (Ne, 4)
    K0: sp.csr_matrix            # stiffness for unit conductivity
    m0: np.ndarray               # lumped volume per node (V/4 summed), multiply by rho*cp
    faces: np.ndarray            # (Nf, 3) boundary triangles on the physical surface
    face_area: np.ndarray
    face_xyz: np.ndarray         # face centroids projected onto the true surface
    face_nrm: np.ndarray         # outward unit normals (from the implicit function)
    surf_nodes: np.ndarray       # (S,) node ids that touch a surface face
    surf_xyz: np.ndarray         # (S, 3)
    surf_nrm: np.ndarray         # (S, 3)
    surf_tag: np.ndarray         # (S,) component index (geometry.TAGS)
    info: dict

    @property
    def n_nodes(self):
        return len(self.nodes)


def _poisson(P, r, fixed=None, order=None, rng=None):
    """Greedy Poisson-disk selection with per-point radius r (small radii first)."""
    n = len(P)
    tree = cKDTree(P)
    blocked = np.zeros(n, bool)
    if fixed is not None and len(fixed):
        d, _ = cKDTree(fixed).query(P)
        blocked |= d < r
    if order is None:
        order = np.argsort(r + 1e-6 * (rng.random(n) if rng is not None else 0))
    acc = []
    for i in order:
        if blocked[i]:
            continue
        acc.append(i)
        blocked[tree.query_ball_point(P[i], r[i])] = True
    return np.array(acc, int)


def _surface_sizing(dom: Domain, P, N, spec: MeshSpec):
    a = np.where(np.abs(N[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(N, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(N, t1)
    delta, kappa = 0.05, np.zeros(len(P))
    for t in (t1, t2):
        for s in (1.0, -1.0):
            Nq = dom.normals(P + s * delta * t)
            ang = np.arccos(np.clip((N * Nq).sum(1), -1, 1))
            kappa = np.maximum(kappa, ang / delta)
    size = np.clip(spec.curv / np.maximum(kappa, 1e-3), spec.h_min, spec.h_max)
    # local wall thickness: march inwards until the solid is left
    thick = np.full(len(P), 0.8)
    alive = np.ones(len(P), bool)
    for step in np.arange(0.02, 0.8, 0.02):
        q = P[alive] - N[alive] * step
        out = dom.implicit(q) > 0
        idx = np.where(alive)[0][out]
        thick[idx] = step
        alive[idx] = False
        if not alive.any():
            break
    size = np.minimum(size, np.clip(3.0 * thick, spec.h_min, spec.h_max))
    # grade: size_i <= size_j + grow * dist(i, j) for the fine points j
    fine = np.where(size < 0.9 * spec.h_max)[0]
    if len(fine):
        k = min(12, len(fine))
        d, j = cKDTree(P[fine]).query(P, k=k)
        d, j = d.reshape(len(P), -1), j.reshape(len(P), -1)
        size = np.minimum(size, (size[fine][j] + spec.grow * d).min(1))
    return size


def _signed_volumes(nodes, tets):
    a, b, c, d = (nodes[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def _surface_triangulation(dom: Domain, spec: MeshSpec, rng):
    """Points on the surface + triangles on the surface. Returns (S, tri, tri_area, tri_xyz, tri_nrm)."""
    lo, hi = dom.lo - 0.2, dom.hi + 0.2
    cand, total = [], 0
    while total < spec.n_cand:
        p = lo + rng.random((600_000, 3)) * (hi - lo)
        p = p[np.abs(dom.implicit(p)) < 0.3]
        p = dom.project(p)
        p = p[np.abs(dom.implicit(p)) < 1e-4]
        cand.append(p)
        total += len(p)
    P = np.concatenate(cand)[: spec.n_cand]
    N = dom.normals(P)
    size = _surface_sizing(dom, P, N, spec)
    keep = _poisson(P, size, rng=rng)
    S, Ssz = P[keep], size[keep]

    # coarse interior points so the Delaunay inside/outside test is well posed
    ci = lo + rng.random((200_000, 3)) * (hi - lo)
    f = dom.implicit(ci)
    ci, f = ci[f < -0.15], f[f < -0.15]
    d_surf, j_surf = cKDTree(S).query(ci)
    r_in = np.maximum(np.minimum(np.clip(1.4 * (-f), 0.3, spec.h_max), Ssz[j_surf] + spec.grow * d_surf), 0.3)
    ik = _poisson(ci, r_in, fixed=S, rng=rng)
    pts = np.concatenate([S, ci[ik]])

    tets = Delaunay(pts, qhull_options="Qbb Qc Qz Q12").simplices
    tets = tets[dom.sdf(pts[tets].mean(1)) < -5e-4]
    loc = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    allf = tets[:, loc].reshape(-1, 3)
    srt = np.sort(allf, axis=1)
    key = (srt[:, 0].astype(np.int64) * len(pts) + srt[:, 1]) * len(pts) + srt[:, 2]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    bf = allf[first[cnt == 1]]
    bf = bf[(bf < len(S)).all(1)]                        # faces made of surface points only
    cr = np.cross(pts[bf[:, 1]] - pts[bf[:, 0]], pts[bf[:, 2]] - pts[bf[:, 0]])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fc = pts[bf].mean(1)
    tol = np.maximum(0.03, 0.25 * np.sqrt(2.0 * area))
    ok = (np.abs(dom.sdf(fc)) < tol) & (area > 1e-8)
    bf, area, fc = bf[ok], area[ok], fc[ok]
    fxyz = dom.project(fc)
    fnrm = dom.normals(fxyz)
    # orient triangles outward
    flip = np.einsum("ij,ij->i", np.cross(S[bf[:, 1]] - S[bf[:, 0]], S[bf[:, 2]] - S[bf[:, 0]]), fnrm) < 0
    bf[flip] = bf[flip][:, [0, 2, 1]]
    return S, bf, area, fxyz, fnrm


def _column_depths(dom: Domain, S, nrm, spec: MeshSpec):
    """Depth of every column (m): as deep as the clearance to the other surfaces allows (<= d_cap)."""
    depth = np.full(len(S), 0.004)
    alive = np.ones(len(S), bool)
    for d in np.arange(0.004, spec.d_cap + 1e-9, 0.004):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        q = S[idx] - d * nrm[idx]
        clr = -dom.sdf(q)
        good = (clr >= 0.6 * d)
        depth[idx[good]] = d
        alive[idx[~good]] = False
    return np.maximum(depth, 0.008)


def _layer_fractions(D, nl, d0):
    """(n, nl) depths with first layer d0, geometric growth r chosen so the column sums to D."""
    n = len(D)
    d0 = np.minimum(d0, D / (nl - 1) * 0.999)
    lo, hi = np.ones(n), np.full(n, 4.0)
    m = np.arange(nl - 1)
    for _ in range(60):
        r = 0.5 * (lo + hi)
        tot = d0 * (np.power(r[:, None], m[None, :]).sum(1))
        big = tot > D
        hi = np.where(big, r, hi)
        lo = np.where(big, lo, r)
    r = 0.5 * (lo + hi)
    steps = d0[:, None] * np.power(r[:, None], m[None, :])
    return np.concatenate([np.zeros((n, 1)), np.cumsum(steps, 1)], axis=1)


def _prism_tets(t, nl):
    """Split every prism (triangle t[:, 0..2] sorted by index, extruded over nl layers) into 3 tets."""
    tets = []
    for j in range(nl - 1):
        top, bot = t * nl + j, t * nl + j + 1
        v0t, v1t, v2t = top[:, 0], top[:, 1], top[:, 2]
        v0b, v1b, v2b = bot[:, 0], bot[:, 1], bot[:, 2]
        tets += [np.stack([v0t, v1t, v2t, v0b], 1), np.stack([v1t, v2t, v0b, v1b], 1),
                 np.stack([v2t, v0b, v1b, v2b], 1)]
    return np.concatenate(tets)


def build_mesh(dom: Domain, spec: MeshSpec, face_filter: Callable | None = None, verbose=False) -> Mesh:
    t0 = time.perf_counter()
    rng = np.random.default_rng(spec.seed)
    S, tri, tri_area, tri_xyz, tri_nrm = _surface_triangulation(dom, spec, rng)
    if face_filter is not None:
        keep_f = face_filter(tri_xyz, tri_nrm)
    else:
        keep_f = np.ones(len(tri), bool)

    # node normals: area-weighted average of adjacent face normals
    nn = np.zeros((len(S), 3))
    for i in range(3):
        np.add.at(nn, tri[:, i], tri_nrm * tri_area[:, None])
    used = np.unique(tri)
    nn_len = np.linalg.norm(nn, axis=1)
    nrm = nn / np.maximum(nn_len, 1e-12)[:, None]
    # nodes without faces (should not happen) fall back to the implicit-function normal
    bad = nn_len < 1e-12
    if bad.any():
        nrm[bad] = dom.normals(S[bad])

    # compact node numbering to the nodes that belong to a face
    remap = -np.ones(len(S), int)
    remap[used] = np.arange(len(used))
    S, nrm, tri = S[used], nrm[used], remap[tri]
    ns, nl = len(S), spec.nl

    D = _column_depths(dom, S, nrm, spec)
    # columns of crease nodes tilt away from the face normals; keep them short enough not to cross neighbours
    ang = np.zeros(len(S))
    emin = np.full(len(S), np.inf)
    for i in range(3):
        c = np.clip((nrm[tri[:, i]] * tri_nrm).sum(1), -1, 1)
        np.maximum.at(ang, tri[:, i], np.arccos(c))
        e = np.linalg.norm(S[tri[:, i]] - S[tri[:, (i + 1) % 3]], axis=1)
        np.minimum.at(emin, tri[:, i], e)
        np.minimum.at(emin, tri[:, (i + 1) % 3], e)
    D = np.minimum(D, 0.4 * emin / np.maximum(np.tan(np.minimum(ang, 1.4)), 0.15))
    D = np.maximum(D, 0.004)
    o = np.argsort(tri, axis=1)
    tsort = np.take_along_axis(tri, o, axis=1)                            # v0 < v1 < v2 (conforming split rule)
    dep = _layer_fractions(D, nl, spec.d0)                                # (ns, nl)
    nodes = (S[:, None, :] - dep[:, :, None] * nrm[:, None, :]).reshape(-1, 3)   # id = i*nl + j
    tets = _prism_tets(tsort, nl)
    vol = _signed_volumes(nodes, tets)
    sg = np.sign(vol).reshape(nl - 1, 3, -1)
    twisted = (sg.min(1) != sg.max(1)) | (np.abs(vol).reshape(nl - 1, 3, -1).min(1) < 1e-14)   # (layer, prism)
    bad_prism = twisted.any(0)                                            # whole column-stack of that triangle is dropped
    n_dropped = int(bad_prism.sum())
    good = np.tile(~bad_prism, (nl - 1) * 3)
    flip = vol < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    vol = np.abs(vol)
    tets, vol = tets[good], vol[good]
    # triangles whose prisms were dropped are not part of the skin: remove them (and their flux area)
    tri_ok = ~bad_prism
    area_total = float(tri_area[keep_f].sum())

    uniq = np.unique(tets)
    nmap = -np.ones(len(nodes), int)
    nmap[uniq] = np.arange(len(uniq))
    nodes_full_ids = uniq
    tets = nmap[tets]
    nodes = nodes[uniq]

    # FEM matrices (unit conductivity) ---------------------------------------------------------
    X = nodes[tets]
    Jm = X[:, 1:] - X[:, :1]
    inv = np.linalg.inv(Jm)
    G = np.zeros((len(tets), 4, 3))
    G[:, 1:, :] = np.transpose(inv, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(1)
    Ke = np.einsum("eij,ekj->eik", G, G) * vol[:, None, None]
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    K0 = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(len(nodes),) * 2).tocsr()
    m0 = np.zeros(len(nodes))
    np.add.at(m0, tets.ravel(), np.repeat(vol / 4.0, 4))

    # boundary faces (wetted triangles) = top layer of each surviving surface triangle ---------------------------
    sel = keep_f & tri_ok
    faces = nmap[tri[sel] * nl]
    area = tri_area[sel]
    fxyz, fnrm = tri_xyz[sel], tri_nrm[sel]
    surf_nodes = np.unique(faces)
    sxyz = nodes[surf_nodes]
    top_ids = nodes_full_ids[surf_nodes] // nl
    snrm = nrm[top_ids]
    stag = geometry.tag_points(sxyz) if dom is AIRCRAFT else np.zeros(len(surf_nodes), np.int8)

    info = dict(n_nodes=len(nodes), n_tets=len(tets), n_surface_nodes=len(surf_nodes), n_faces=len(faces),
                skin_volume=float(vol.sum()), surface_area=float(area.sum()),
                mean_depth=float(D.mean()), min_depth=float(D.min()), max_depth=float(D.max()),
                prisms_dropped=n_dropped, area_kept_fraction=float(tri_area[sel].sum() / area_total), n_prisms=int((nl - 1) * len(tri)), build_s=time.perf_counter() - t0)
    if verbose:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
    return Mesh(nodes, tets, K0, m0, faces, area, fxyz, fnrm, surf_nodes, sxyz, snrm, stag, info)


_CACHE: dict = {}


def aircraft_mesh(name: str = "coarse") -> Mesh:
    """Build (once per process) the aircraft mesh of the given resolution."""
    if name not in _CACHE:
        _CACHE[name] = build_mesh(AIRCRAFT, SPECS[name])
    return _CACHE[name]
