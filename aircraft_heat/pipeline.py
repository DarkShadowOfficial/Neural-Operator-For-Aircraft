"""End-to-end pipeline: FEM data -> train an ensemble -> held-out evaluation -> scaling benchmark."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import analysis, complexity, fem
from .config import Preset
from .data import build_dataset
from .mesh import aircraft_mesh
from .operator import (DeepONet, Ensemble, drive_scale, initial_field, make_fieldset, static_features, train_model)


def evaluate(ens: Ensemble, P, T_true, times, static, xyz, tags, key="full") -> dict:
    """Metrics of the ensemble against FEM truth over a set of conditions (one entry per condition)."""
    per, all_err, all_base, all_std = [], [], [], []
    comp = None
    for i in range(len(P)):
        mean, std = ens.predict(P[i], times, key, static, xyz)
        ic = initial_field(P[i:i + 1], xyz)[0]
        err = mean - T_true[i]
        m = analysis.error_metrics(mean, T_true[i], baseline=np.broadcast_to(ic, T_true[i].shape), std=std)
        per.append(m)
        all_err.append(err)
        all_base.append(np.broadcast_to(ic, T_true[i].shape) - T_true[i])
        all_std.append(std)
    E, B, Sd = np.array(all_err), np.array(all_base), np.array(all_std)
    agg = dict(mae=float(np.abs(E).mean()), rmse=float(np.sqrt((E ** 2).mean())), max=float(np.abs(E).max()),
               baseline_mae=float(np.abs(B).mean()), baseline_rmse=float(np.sqrt((B ** 2).mean())),
               skill=float(1 - (E ** 2).mean() / (B ** 2).mean()),
               rel_rmse_pct=float(100 * np.sqrt((E ** 2).mean() / (B ** 2).mean())),
               mean_std=float(Sd.mean()), coverage_2sigma=float(np.mean(np.abs(E) <= 2 * Sd + 1e-9)),
               corr_std_err=float(np.corrcoef(Sd.ravel(), np.abs(E).ravel())[0, 1]))
    agg["by_time"] = analysis.by_time_bin(E.transpose(1, 0, 2), times)
    agg["by_component"] = analysis.by_component(np.abs(E).mean(0), tags)
    agg["per_case_mae"] = [m["mae"] for m in per]
    agg["per_case_rel_rmse_pct"] = [m["rel_rmse_pct"] for m in per]
    agg["per_case_max"] = [m["max"] for m in per]
    return agg


def node_weights(mesh) -> np.ndarray:
    """Loss weights per surface node: half uniform, half proportional to the nodal surface-area share."""
    area = np.zeros(mesh.n_nodes)
    for i in range(3):
        np.add.at(area, mesh.faces[:, i], mesh.face_area / 3)
    return 0.5 + 0.5 * area[mesh.surf_nodes] / max(area[mesh.surf_nodes].mean(), 1e-12)


class BudgetExceeded(Exception):
    """Raised between work units when `budget_s` is used up; progress is cached, call again to continue."""


def run_pipeline(cfg: Preset, progress, ckpt_dir=None, budget_s: float | None = None) -> dict:
    t_all = time.time()

    def check_budget():
        if budget_s is not None and time.time() - t_all > budget_s:
            raise BudgetExceeded()
    mdir = None if ckpt_dir is None else Path(ckpt_dir) / "members"
    if mdir is not None:
        mdir.mkdir(parents=True, exist_ok=True)
    progress.set_stage("meshing", f"building the {cfg.mesh} FEM mesh and boundary-condition geometry")
    mesh = aircraft_mesh(cfg.mesh)
    solver = fem.get_solver(cfg.mesh)
    xyz, nrm, tags = mesh.surf_xyz, mesh.surf_nrm, mesh.surf_tag
    static = static_features(xyz, nrm, tags)
    w_nodes = node_weights(mesh)

    progress.set_stage("fem-data", f"running {cfg.n_train + cfg.n_val + cfg.n_test} FEM scenarios (train/val/test)")
    t0 = time.time()
    ds = build_dataset(cfg, ckpt_dir, progress)
    t_data = time.time() - t0
    times = ds["times"]
    tr = make_fieldset(ds["train_c"], times, ds["train_T"], xyz)
    va = make_fieldset(ds["val_c"], times, ds["val_T"], xyz)

    progress.set_stage("training", f"{cfg.ensemble} operators x {cfg.epochs} steps")
    models, hists, train_s = [], [], []
    for i in range(cfg.ensemble):
        mcfg = dict(width=cfg.width, latent=cfg.latent, n_fourier=cfg.n_fourier, seed=cfg.seed * 100 + i)
        f = None if mdir is None else mdir / f"member_{cfg.name}_{i}.npz"
        if f is not None and f.exists():
            z = np.load(f)
            m = DeepONet.from_state(DeepONet(**mcfg).cfg, {k: z[k] for k in z.files if k not in ("hist", "secs")})
            h, secs = z["hist"].tolist(), float(z["secs"])
            progress.add_log(f"operator {i + 1}/{cfg.ensemble}: loaded from cache")
        else:
            check_budget()
            m = DeepONet(**mcfg)
            m, h, secs = train_model(m, tr, va, static, w_nodes, cfg, seed=i, progress=progress, label=f"operator {i + 1}/{cfg.ensemble}")
            if f is not None:
                np.savez(f, hist=np.array(h), secs=secs, **m.state())
            progress.add_log(f"operator {i + 1}/{cfg.ensemble} trained in {secs:.0f} s (val rmse {h[-1][2]:.4f})")
        models.append(m)
        hists.append(h)
        train_s.append(secs)
    ens = Ensemble(models)

    progress.set_stage("evaluating", "held-out scenarios the operator never saw")
    ev_test = evaluate(ens, P=ds["test_c"], times=times, static=static, T_true=ds["test_T"], xyz=xyz, tags=tags)
    ev_val = evaluate(ens, P=ds["val_c"], times=times, static=static, T_true=ds["val_T"], xyz=xyz, tags=tags)
    sub = np.arange(min(len(ds["train_c"]), 16))
    ev_train = evaluate(ens, P=ds["train_c"][sub], times=times, static=static, T_true=ds["train_T"][sub], xyz=xyz, tags=tags)
    progress.add_log(f"test MAE {ev_test['mae']:.2f} K (train {ev_train['mae']:.2f} K), skill {ev_test['skill']:.3f}")

    # learning curve: does error keep falling with more FEM runs, or has it saturated / is it memorising?
    progress.set_stage("learning-curve", "training on fewer FEM runs to check data efficiency")
    curve = []
    lc_file = None if mdir is None else mdir / f"learning_curve_{cfg.name}.json"
    cached = json.loads(lc_file.read_text()) if lc_file is not None and lc_file.exists() else {}
    fracs = (0.25, 0.5) if cfg.n_train >= 16 else ()
    for f_ in fracs:
        n = max(4, int(round(f_ * cfg.n_train)))
        if str(n) in cached:
            curve.append(cached[str(n)])
            continue
        check_budget()
        sub_tr = make_fieldset(ds["train_c"][:n], times, ds["train_T"][:n], xyz)
        m = DeepONet(cfg.width, cfg.latent, cfg.n_fourier, seed=777)
        m, _, _ = train_model(m, sub_tr, va, static, w_nodes, cfg, epochs=max(300, cfg.epochs // 2), seed=5,
                              progress=progress, label=f"learning curve ({n} runs)")
        e1 = evaluate(Ensemble([m]), P=ds["test_c"], times=times, static=static, T_true=ds["test_T"], xyz=xyz, tags=tags, key=f"lc{n}")
        e0 = evaluate(Ensemble([m]), P=ds["train_c"][:n], times=times, static=static, T_true=ds["train_T"][:n], xyz=xyz, tags=tags, key=f"lc{n}")
        curve.append(dict(n_runs=n, test_mae=e1["mae"], train_mae=e0["mae"], test_rel_rmse_pct=e1["rel_rmse_pct"]))
        if lc_file is not None:
            cached[str(n)] = curve[-1]
            lc_file.write_text(json.dumps(cached))
    curve.append(dict(n_runs=cfg.n_train, test_mae=ev_test["mae"], train_mae=ev_train["mae"],
                      test_rel_rmse_pct=ev_test["rel_rmse_pct"], ensemble=True))

    bundle = dict(model_cfgs=[m.cfg for m in models], model_states=[m.state() for m in models])
    meta = dict(preset=cfg.name, mesh=cfg.mesh, n_surface=int(len(xyz)), n_nodes=int(mesh.n_nodes),
                config={k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                mesh_info={k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in mesh.info.items()},
                eval=dict(test=ev_test, val=ev_val, train=ev_train), learning_curve=curve,
                timings=dict(fem_data_s=float(np.sum(np.concatenate([ds["train_wall"], ds["val_wall"], ds["test_wall"]]))), train_s=[float(s) for s in train_s],
                             fem_mean_s=float(np.mean(np.concatenate([ds["train_wall"], ds["val_wall"], ds["test_wall"]]))),
                             total_s=float(time.time() - t_all)),
                history=[[list(map(float, r)) for r in h] for h in hists], trained_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    bundle["meta"] = meta

    progress.set_stage("benchmark", "measuring how FEM and operator cost scale")
    from .inference import OperatorService
    svc = OperatorService(bundle)
    meshes = ("coarse",) if cfg.name == "smoke" else ("coarse", "medium")
    meta["scaling"] = dict(fem=complexity.fem_scaling(meshes, progress), operator=complexity.operator_scaling(svc))
    meta["timings"]["total_s"] = float(time.time() - t_all)
    return bundle


def load_adapted(ckpt_dir, cfg: Preset):
    """Previously adapted bundle (weights + adaptation history) if one exists."""
    from .persistence import load_bundle
    return load_bundle(Path(ckpt_dir) / "adapted", cfg)


def run_adaptation(bundle: dict, cfg: Preset, progress, ckpt_dir, **opts) -> dict:
    """Error-driven fine-tuning on fresh FEM scenarios until the accuracy target is met (see adapt.py)."""
    from .adapt import adapt_bundle
    mesh = aircraft_mesh(cfg.mesh)
    fem.get_solver(cfg.mesh)
    xyz, nrm, tags = mesh.surf_xyz, mesh.surf_nrm, mesh.surf_tag
    static = static_features(xyz, nrm, tags)
    ds = build_dataset(cfg, ckpt_dir, progress)           # original data: replay buffer + untouched test set

    def test_eval(models):
        e = evaluate(Ensemble(models), P=ds["test_c"], times=ds["times"], static=static, T_true=ds["test_T"],
                     xyz=xyz, tags=tags)
        return {k: e[k] for k in ("mae", "rmse", "max", "rel_rmse_pct", "skill", "coverage_2sigma")}
    return adapt_bundle(bundle, cfg, ds, static, xyz, node_weights(mesh), progress, ckpt_dir, test_eval=test_eval, **opts)
