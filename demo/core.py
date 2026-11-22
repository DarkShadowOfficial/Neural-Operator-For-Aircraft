"""Framework-independent demo logic (used by the FastAPI and the stdlib servers).

On start-up the app loads a saved checkpoint or generates FEM data and trains the operator in a background
thread (progress is exposed by ``status()``).  Once ready, ``simulate`` / ``snapshot`` / ``stress`` compare the
operator with the built-in FEM for any flight condition, material and initial temperature.
"""
from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

import numpy as np

from aircraft_heat import analysis, config, fem, geometry
from aircraft_heat.config import (MACH_MAX, MACH_MIN, MATERIALS, MAX_TIME, P_DEFAULT, PARAMS, PNAMES, get_preset,
                                  mach_number)
from aircraft_heat.inference import OperatorService
from aircraft_heat.operator import initial_field
from aircraft_heat.persistence import load_bundle, save_bundle
from aircraft_heat.pipeline import load_adapted, run_adaptation, run_pipeline
from aircraft_heat.progress import Progress

log = logging.getLogger("aircraft_heat.demo")

# hard physical limits for requests (the operator only claims accuracy inside config.PARAMS ranges)
LIMITS = dict(velocity=(20.0, 2500.0), T_air=(150.0, 1000.0), altitude=(0.0, 20000.0), T_init=(150.0, 1500.0),
              dT_axial=(-300.0, 300.0), k=(0.1, 500.0), rho=(300.0, 20000.0), cp=(100.0, 3000.0))


def _r(a, nd=2):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def parse_request(body: dict) -> tuple[dict, dict]:
    """Validate a request body -> (conditions dict, options dict).  Raises ValueError with a readable message."""
    b = dict(body or {})
    mat = b.get("material")
    if mat and mat in MATERIALS:
        for k, v in MATERIALS[mat].items():
            b.setdefault(k, v)
    cond = {}
    for n in PNAMES:
        v = b.get(n, P_DEFAULT[n])
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{n} must be a number")
        lo, hi = LIMITS[n]
        if not np.isfinite(v) or not (lo <= v <= hi):
            raise ValueError(f"{n} must be between {lo:g} and {hi:g}")
        cond[n] = v
    M = float(mach_number(cond["velocity"], cond["T_air"]))
    if not (MACH_MIN <= M <= MACH_MAX):
        raise ValueError(f"Mach {M:.2f} is outside the validity range {MACH_MIN}-{MACH_MAX} of the aerodynamic-heating "
                         f"model; change flight speed or air temperature (speed of sound = {np.sqrt(1.4 * 287.05 * cond['T_air']):.0f} m/s)")
    opt = dict(t=float(b.get("t", 60.0)), n_times=int(b.get("n_times", 61)), n_cases=int(b.get("n_cases", 6)),
               seed=b.get("seed"))
    if not (0.0 <= opt["t"] <= MAX_TIME):
        raise ValueError(f"t must be between 0 and {MAX_TIME:g} s")
    if not (11 <= opt["n_times"] <= 121):
        raise ValueError("n_times must be between 11 and 121")
    if not (2 <= opt["n_cases"] <= 12):
        raise ValueError("n_cases must be between 2 and 12")
    return cond, opt


class DemoApp:
    def __init__(self, preset="quick", ckpt_root="checkpoints", retrain=False, adapt: dict | None = None):
        """adapt: None disables error-driven fine-tuning at start-up; otherwise options for pipeline.run_adaptation
        (target, worst_factor, budget_s, n_probe, N, ...)."""
        self.adapt = adapt
        self.preset_name = preset
        self.cfg = get_preset(preset)
        self.ckpt_dir = Path(ckpt_root) / preset
        self.retrain = retrain
        self.state = "loading"
        self.error: str | None = None
        self.from_checkpoint = False
        self.progress = Progress()
        self.service: OperatorService | None = None
        self.solver = None
        self.lock = threading.Lock()          # serialises compute so timings are clean

    # ------------------------------------------------------------------ boot
    def start(self):
        threading.Thread(target=self.boot, daemon=True, name="trainer").start()

    def boot(self):
        try:
            bundle = None if self.retrain else load_bundle(self.ckpt_dir, self.cfg)
            if bundle is not None:
                self.progress.set_stage("loading", f"loaded checkpoint from {self.ckpt_dir}")
                self.from_checkpoint = True
                self.progress.set_stage("meshing", "building the FEM mesh (one-off, a few seconds)")
            else:
                self.state = "training"
                bundle = run_pipeline(self.cfg, self.progress, self.ckpt_dir)
                save_bundle(bundle, self.ckpt_dir)
                self.progress.add_log(f"saved checkpoint to {self.ckpt_dir}")
            if self.adapt is not None:
                prev = None if self.retrain else load_adapted(self.ckpt_dir, self.cfg)
                if prev is not None:
                    bundle = prev
                if not (bundle["meta"].get("adaptation") or {}).get("reached"):
                    self.state = "adapting"
                    bundle = run_adaptation(bundle, self.cfg, self.progress, self.ckpt_dir, **self.adapt)
            self.service = OperatorService(bundle)
            self.solver = fem.get_solver(self.cfg.mesh)
            self._warmup()
            self.state = "ready"
            self.progress.set_stage("ready", "")
        except Exception as exc:                                   # pragma: no cover
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("boot failed\n%s", traceback.format_exc())

    def _warmup(self):
        c = dict(P_DEFAULT)
        c["velocity"], c["T_air"] = 900.0, 350.0
        self.service.predict_surface(c, np.linspace(0, MAX_TIME, 11))
        self.solver.solve(c, [0.0, 10.0])

    def require_ready(self):
        if self.state != "ready":
            raise RuntimeError(f"operator not ready (state={self.state})")

    # ------------------------------------------------------------------ meta
    def status(self):
        return {"state": self.state, "preset": self.preset_name, "error": self.error,
                "from_checkpoint": self.from_checkpoint, "progress": self.progress.snapshot()}

    def config(self):
        out = {
            "params": [dict(name=p[0], label=p[1], unit=p[2], lo=p[3], hi=p[4], log=bool(p[5]), default=p[6]) for p in PARAMS],
            "materials": MATERIALS, "limits": LIMITS, "max_time": MAX_TIME, "mach_range": [MACH_MIN, MACH_MAX],
            "preset": self.preset_name, "ready": self.state == "ready", "training": None,
            "tags": geometry.TAGS, "probe_names": [p[0] for p in geometry.PROBES],
            "time_bins": [list(b) for b in analysis.TIME_BINS],
        }
        if self.service is not None:
            s = self.service
            out["surface"] = dict(xyz=_r(s.xyz, 3), tag=s.tag.tolist())
            out["probes"] = s.probes
            out["training"] = s.meta
        return out

    def random_case(self, seed=None):
        rng = np.random.default_rng(seed)
        v = config.sample_conditions(1, rng)[0]
        return {n: float(x) for n, x in zip(PNAMES, v)}

    # ------------------------------------------------------------------ compute
    def _compare(self, cond, times):
        svc, solver = self.service, self.solver
        r = solver.solve(cond, times)
        fem_s = r.info["wall_s"]
        svc.predict_surface(cond, times)                                  # warm
        (op, std), op_s = analysis.timed(lambda: svc.predict_surface(cond, times), 5)
        return r, fem_s, op, std, op_s

    def simulate(self, body):
        self.require_ready()
        cond, opt = parse_request(body)
        with self.lock:
            svc = self.service
            times = np.linspace(0.0, MAX_TIME, opt["n_times"])
            r, fem_s, op, std, op_s = self._compare(cond, times)
            ref = r.T_surf
            err = op - ref
            P = np.array([[cond[n] for n in PNAMES]])
            ic = np.broadcast_to(initial_field(P, svc.xyz)[0], ref.shape)
            metrics = analysis.error_metrics(op, ref, baseline=ic, std=std)
            probes = []
            for p in svc.probes:
                j = p["node"]
                probes.append(dict(name=p["name"], node=j, xyz=_r(svc.xyz[j], 2), tag=geometry.TAGS[int(svc.tag[j])],
                                   fem=_r(ref[:, j], 1), op=_r(op[:, j], 1), std=_r(std[:, j], 2), err=_r(err[:, j], 2),
                                   t_ic=float(ic[0, j])))
            k = int(np.argmin(np.abs(times - opt["t"])))
            return {
                "conditions": cond, "mach": float(mach_number(cond["velocity"], cond["T_air"])),
                "warnings": svc.check(cond),
                "times": times.tolist(), "n_surface": svc.n_surface,
                "fem": _r(ref, 1), "op": _r(op, 1), "err": _r(err, 2), "std": _r(std, 2),
                "ic": _r(ic[0], 1), "probes": probes, "t_index": k,
                "metrics": metrics,
                "by_time": analysis.by_time_bin(err, times),
                "by_component": analysis.by_component(np.abs(err), svc.tag),
                "max_error_at": analysis.locate_max(err, times, svc.xyz),
                "mae_vs_time": _r(np.abs(err).mean(1), 3), "max_vs_time": _r(np.abs(err).max(1), 3),
                "aero": r.info["aero"],
                "timing": dict(fem_s=fem_s, op_s=op_s, speedup=fem_s / op_s if op_s > 0 else None,
                               fem_steps=r.info["steps"], fem_nodes=r.info["n_nodes"], fem_factor_s=r.info["setup_s"],
                               grid_points=int(len(times) * svc.n_surface)),
            }

    def snapshot(self, body):
        """Cost of ONE requested time: the FEM must integrate from 0 to t, the operator just evaluates at t."""
        self.require_ready()
        cond, opt = parse_request(body)
        t = max(opt["t"], 0.5)
        with self.lock:
            svc = self.service
            r, fem_s, op, std, op_s = self._compare(cond, [t])
            err = op[0] - r.T_surf[0]
            return {"t": t, "fem_s": fem_s, "op_s": op_s, "speedup": fem_s / op_s, "fem_steps": r.info["steps"],
                    "mae": float(np.abs(err).mean()), "max": float(np.abs(err).max())}

    def stress(self, body):
        """Fresh random scenarios (never in training) -> distribution of operator error and speed-up."""
        self.require_ready()
        _, opt = parse_request(body if body else {})
        n = opt["n_cases"]
        rng = np.random.default_rng(opt["seed"])
        with self.lock:
            svc = self.service
            times = np.linspace(0.0, MAX_TIME, 41)
            rows = []
            for vec in config.sample_conditions(n, rng):
                cond = {nm: float(x) for nm, x in zip(PNAMES, vec)}
                r, fem_s, op, std, op_s = self._compare(cond, times)
                P = np.array([[cond[nm] for nm in PNAMES]])
                ic = np.broadcast_to(initial_field(P, svc.xyz)[0], r.T_surf.shape)
                m = analysis.error_metrics(op, r.T_surf, baseline=ic, std=std)
                rows.append(dict(conditions=cond, mach=float(mach_number(cond["velocity"], cond["T_air"])), mae=m["mae"],
                                 rmse=m["rmse"], max=m["max"], rel_rmse_pct=m["rel_rmse_pct"], skill=m["skill"],
                                 fem_s=fem_s, op_s=op_s))
            mae = np.array([x["mae"] for x in rows])
            return dict(cases=rows, mean_mae=float(mae.mean()), worst_mae=float(mae.max()),
                        mean_rel_rmse_pct=float(np.mean([x["rel_rmse_pct"] for x in rows])),
                        mean_speedup=float(np.mean([x["fem_s"] / x["op_s"] for x in rows])))
