"""FEM dataset generation: the operator is trained on (and judged against) unlimited fresh FEM runs."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from . import config
from .config import MAX_TIME, PNAMES, Preset


def cond_dict(vec) -> dict:
    return {n: float(v) for n, v in zip(PNAMES, vec)}


_W = {}


def _init_worker(mesh_name):
    from .fem import get_solver
    _W["s"] = get_solver(mesh_name)


def _run_one(args):
    vec, times = args
    r = _W["s"].solve(cond_dict(vec), times)
    return r.T_surf.astype(np.float32), r.info["wall_s"]


def run_fem_batch(vecs: np.ndarray, times: np.ndarray, mesh_name: str, progress=None, label="FEM runs", workers=None):
    """Run the FEM for every condition; returns (T[n, nt, S] float32, wall seconds per run)."""
    workers = workers or max(1, (os.cpu_count() or 1))
    jobs = [(v, times) for v in vecs]
    out, walls = [], []
    t0 = time.time()
    if workers > 1 and len(jobs) > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        from .fem import get_solver
        get_solver(mesh_name)                                    # build once so forked workers inherit it
        with ctx.Pool(workers, initializer=_init_worker, initargs=(mesh_name,)) as pool:
            for i, (T, w) in enumerate(pool.imap(_run_one, jobs)):
                out.append(T)
                walls.append(w)
                if progress:
                    progress.update((i + 1) / len(jobs), f"{label}: {i + 1}/{len(jobs)}")
    else:
        _init_worker(mesh_name)
        for i, j in enumerate(jobs):
            T, w = _run_one(j)
            out.append(T)
            walls.append(w)
            if progress:
                progress.update((i + 1) / len(jobs), f"{label}: {i + 1}/{len(jobs)}  ({(time.time() - t0) / (i + 1):.1f} s each)")
    return np.array(out), np.array(walls)


def build_dataset(cfg: Preset, cache_dir: Path | None = None, progress=None) -> dict:
    """train / val / test FEM runs (independent random draws of the condition space)."""
    cache = None if cache_dir is None else Path(cache_dir) / "dataset.npz"
    key = np.array([cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_times, cfg.seed])
    if cache is not None and cache.exists():
        d = np.load(cache, allow_pickle=False)
        if d["mesh"].item() == cfg.mesh and np.array_equal(d["key"], key):
            if progress:
                progress.add_log(f"loaded FEM dataset from {cache}")
            return {k: d[k] for k in d.files}
    times = np.linspace(0.0, MAX_TIME, cfg.n_times)
    out = dict(times=times, key=key, mesh=np.array(cfg.mesh))
    for name, n, seed in (("train", cfg.n_train, cfg.seed), ("val", cfg.n_val, cfg.seed + 101),
                          ("test", cfg.n_test, cfg.seed + 202)):
        vecs = config.sample_conditions(n, np.random.default_rng(seed))
        T, walls = run_fem_batch(vecs, times, cfg.mesh, progress, label=f"FEM {name}")
        out[f"{name}_c"], out[f"{name}_T"], out[f"{name}_wall"] = vecs, T, walls
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out
