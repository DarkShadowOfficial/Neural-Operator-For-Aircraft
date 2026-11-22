"""Error-driven adaptive fine-tuning (active learning with the built-in FEM as the oracle).

Each round:
  1. draw fresh random scenarios (never used for training) and run FEM + operator on them;
  2. a probe set PASSES if mean per-case MAE <= target AND worst case <= worst_factor*target; stop after
     `required_passes` consecutive passes on different fresh sets (one lucky draw of a few noisy scenarios is not
     evidence); a passing round costs only the probe FEM runs, no training;
  3. otherwise give every scenario a sample budget that grows with its error
         n_i = 0                         if MAE_i < lo           (default 2 K: good enough, no extra data)
         n_i = round(N * MAE_i / ref)    otherwise, at least 1   (ref = 10 K: a 10 K case gets N, a 20 K case 2N, ...)
         capped at cap*N
     the scenario itself is one of its n_i samples (its FEM result already exists, so it is free); the other
     n_i - 1 samples are FEM runs at random neighbours of the scenario in parameter space;
  4. fine-tune every ensemble member on  (original training set + all adaptive samples), oversampling the adaptive
     ones, starting from the current weights (validation can never get worse: the start weights are the first
     "best" candidate);
  5. checkpoint, repeat with a new fresh probe set.

The probe scenarios of a round are only used for training AFTER they were used for measuring, and the stopping test
always uses a NEW probe set, so reaching the target is not caused by memorising the probe set.  The original
held-out test set is never trained on; it is evaluated before the first and after the last round for the record.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from . import config
from .config import Preset
from .data import run_fem_batch
from .operator import DeepONet, Ensemble, FieldSet, initial_field, make_fieldset, train_model


def case_mae(ens: Ensemble, P, T_true, times, static, xyz, key) -> np.ndarray:
    """Mean |error| (K) of the ensemble for every scenario."""
    out = []
    for i in range(len(P)):
        mean, _ = ens.predict(P[i], times, key, static, xyz)
        out.append(float(np.abs(mean - T_true[i]).mean()))
    return np.array(out)


def allocate(errs, N=3, lo=2.0, ref=10.0, cap=4) -> np.ndarray:
    """Samples per scenario from its error: 0 below `lo`, else N*err/ref (>= 1, <= cap*N)."""
    errs = np.asarray(errs, float)
    n = np.clip(np.round(N * errs / ref), 1, cap * N).astype(int)
    return np.where(errs < lo, 0, n)


def neighbours(vec, n, rng, sigma=0.07):
    """n valid scenarios near `vec` (Gaussian jitter in the unit-scaled parameter space)."""
    out, u0 = [], config.to_unit(vec)
    tries = 0
    while len(out) < n and tries < 200:
        tries += 1
        u = np.clip(u0 + rng.normal(0.0, sigma, u0.shape), 0.0, 1.0)
        v = config.from_unit(u)
        if config.MACH_MIN <= float(config.mach_number(v[0], v[1])) <= config.MACH_MAX:
            out.append(v)
    return np.array(out).reshape(-1, config.N_PARAMS)


def _concat(a: FieldSet, b: FieldSet) -> FieldSet:
    return FieldSet(np.concatenate([a.P, b.P]), a.times, np.concatenate([a.Y, b.Y]), np.concatenate([a.T_ic, b.T_ic]),
                    np.concatenate([a.scale, b.scale]), np.concatenate([a.extra, b.extra]))


def _save(ad_dir: Path, models, meta, extra):
    from .persistence import save_bundle
    ad_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(dict(model_cfgs=[m.cfg for m in models], model_states=[m.state() for m in models], meta=meta), ad_dir)
    np.savez_compressed(ad_dir / "extra.npz", P=extra["P"], T=extra["T"], V=extra["V"])


def adapt_bundle(bundle: dict, cfg: Preset, ds: dict, static, xyz, w_nodes, progress, ckpt_dir,
                 target: float = 10.0, worst_factor: float = 2.0, budget_s: float = 900.0, n_probe: int = 10,
                 required_passes: int = 2,
                 N: int = 3, boost: float = 3.0, ft_lr: float = 6e-4, max_rounds: int = 12, seed: int = 4242,
                 test_eval=None) -> dict:
    """Run the adaptive loop and return the improved bundle (also saved to <ckpt_dir>/adapted).

    test_eval(models) -> metrics dict on the untouched held-out test set (for the before/after record)."""
    t0 = time.time()
    ckpt_dir = Path(ckpt_dir)
    ad_dir = ckpt_dir / "adapted"
    times = ds["times"]
    tr = make_fieldset(ds["train_c"], times, ds["train_T"], xyz)
    va = make_fieldset(ds["val_c"], times, ds["val_T"], xyz)
    models = [DeepONet.from_state(c, s) for c, s in zip(bundle["model_cfgs"], bundle["model_states"])]
    ad = dict(bundle["meta"].get("adaptation") or {})
    hist = ad.get("rounds", [])
    extra = dict(P=np.zeros((0, config.N_PARAMS)), T=np.zeros((0,) + ds["train_T"].shape[1:], np.float32), V=np.zeros(0, bool))
    if (ad_dir / "extra.npz").exists() and hist:
        z = np.load(ad_dir / "extra.npz")
        extra = dict(P=z["P"], T=z["T"], V=z["V"] if "V" in z.files else np.zeros(len(z["P"]), bool))
    ad.setdefault("target_mae", target)
    ad["target_mae"], ad["worst_factor"] = target, worst_factor
    if "test_before" not in ad and test_eval is not None:
        ad["test_before"] = test_eval(models)
    reached = bool(ad.get("reached", False))
    passes = int(ad.get("passes", 0))
    stop = "target already reached" if reached else None

    for r in range(len(hist), max_rounds):
        if reached:
            break
        if time.time() - t0 > budget_s:
            stop = "time budget used up"
            break
        rng = np.random.default_rng(seed + 31 * r)
        # 1. fresh probe scenarios -------------------------------------------------------------
        progress.set_stage("adapting", f"round {r + 1}: running FEM on {n_probe} fresh scenarios to measure the error")
        pv = config.sample_conditions(n_probe, rng)
        pT, _ = run_fem_batch(pv, times, cfg.mesh, progress, label=f"round {r + 1} probe FEM")
        ens = Ensemble(models)
        errs = case_mae(ens, pv, pT, times, static, xyz, key=f"adapt{r}")
        mean, worst = float(errs.mean()), float(errs.max())
        rec = dict(round=r + 1, probe_mean_mae=mean, probe_worst_mae=worst, probe_errors=errs.tolist(),
                   n_new=0, n_extra_total=int(len(extra["P"])), elapsed_s=time.time() - t0)
        progress.add_log(f"adapt round {r + 1}: fresh-scenario MAE mean {mean:.1f} K, worst {worst:.1f} K "
                         f"(target mean <= {target:g}, worst <= {worst_factor * target:g})")
        # 2. stopping rule: required_passes consecutive passing fresh probe sets ---------------------------
        ok = mean <= target and worst <= worst_factor * target
        passes = passes + 1 if ok else 0
        rec["passed"] = ok
        if passes >= required_passes:
            reached = True
            rec["reached"] = True
            hist.append(rec)
            stop = f"target reached on {required_passes} consecutive fresh unseen scenario sets"
            break
        if ok:                                           # passed once: measure again on a NEW set, no training
            hist.append(rec)
            ad.update(rounds=hist, reached=False, passes=passes, n_extra=int(len(extra["P"])))
            _save(ad_dir, models, dict(bundle["meta"], adaptation=ad), extra)
            continue
        # 3. error-proportional sample allocation ---------------------------------------------------
        alloc = allocate(errs, N=N)
        vecs, Ts = [], []
        todo = []
        for i, n_i in enumerate(alloc):
            if n_i == 0:
                continue
            vecs.append(pv[i])
            Ts.append(pT[i])                                   # the case itself: FEM result already exists
            if n_i > 1:
                todo.extend(neighbours(pv[i], int(n_i) - 1, rng))
        if todo:
            progress.set_stage("adapting", f"round {r + 1}: {len(todo)} extra FEM runs around the hard cases")
            nT, _ = run_fem_batch(np.array(todo), times, cfg.mesh, progress, label=f"round {r + 1} neighbour FEM")
            vecs.extend(todo)
            Ts.extend(list(nT))
        rec["allocation"] = alloc.tolist()
        rec["n_new"] = int(len(vecs))
        if vecs:
            isv = rng.random(len(vecs)) < 0.25                    # 25 % held back as adaptive VALIDATION (never trained on)
            extra = dict(P=np.concatenate([extra["P"], np.array(vecs)]),
                         T=np.concatenate([extra["T"], np.array(Ts, np.float32)]),
                         V=np.concatenate([extra["V"], isv]))
            # 4. fine-tune on original + adaptive samples (adaptive ones oversampled) -----------------
            tm, vm = ~extra["V"], extra["V"]
            ex = make_fieldset(extra["P"][tm], times, extra["T"][tm], xyz)
            both = _concat(tr, ex)
            va_all = _concat(va, make_fieldset(extra["P"][vm], times, extra["T"][vm], xyz)) if vm.any() else va
            p = np.concatenate([np.ones(len(tr.P)), np.full(len(ex.P), boost)])
            p /= p.sum()
            steps = int(np.clip(600 + 40 * int(tm.sum()), 800, 3500))
            fcfg = dataclasses.replace(cfg, lr=ft_lr)
            for i, m in enumerate(models):
                models[i], _, secs = train_model(m, both, va_all, static, w_nodes, fcfg, epochs=steps, seed=r * 10 + i,
                                                 progress=progress, label=f"round {r + 1} fine-tune {i + 1}/{len(models)}",
                                                 log_every=max(200, steps // 4), cond_probs=p, warm_start=True)
        rec["elapsed_s"] = time.time() - t0
        hist.append(rec)
        # 5. checkpoint every round ---------------------------------------------------------------------
        ad.update(rounds=hist, reached=False, passes=0, n_extra=int(len(extra["P"])))
        _save(ad_dir, models, dict(bundle["meta"], adaptation=ad), extra)
    else:
        stop = stop or "maximum rounds"

    ad.update(rounds=hist, reached=reached, passes=passes, required_passes=required_passes, n_extra=int(len(extra["P"])), stop_reason=stop or "time budget used up",
              elapsed_s=float(ad.get("elapsed_s", 0.0) + time.time() - t0))
    if test_eval is not None:
        ad["test_after"] = test_eval(models)
    meta = dict(bundle["meta"], adaptation=ad)
    out = dict(model_cfgs=[m.cfg for m in models], model_states=[m.state() for m in models], meta=meta)
    _save(ad_dir, models, meta, extra)
    progress.add_log(f"adaptation finished: {ad['stop_reason']}; {len(hist)} rounds, {ad['n_extra']} adaptive samples")
    return out
