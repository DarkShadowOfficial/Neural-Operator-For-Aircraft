"""Error metrics, regional breakdowns and timing helpers shared by training reports and the demo."""
from __future__ import annotations

import time

import numpy as np

from . import geometry

TIME_BINS = [(0.0, 20.0), (20.0, 60.0), (60.0, 120.0)]


def timed(fn, repeat: int = 1):
    """(result, best wall seconds over `repeat` calls)."""
    best, out = np.inf, None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def error_metrics(pred, ref, baseline=None, std=None) -> dict:
    """Metrics (kelvin) over arrays of identical shape. `baseline` = 'nothing happens' guess (initial T)."""
    e = np.asarray(pred, float) - np.asarray(ref, float)
    out = dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
               max=float(np.max(np.abs(e))), bias=float(np.mean(e)),
               p95=float(np.percentile(np.abs(e), 95)))
    if baseline is not None:
        e0 = np.asarray(baseline, float) - np.asarray(ref, float)
        out["baseline_mae"] = float(np.mean(np.abs(e0)))
        out["baseline_rmse"] = float(np.sqrt(np.mean(e0 ** 2)))
        out["skill"] = float(1.0 - np.mean(e ** 2) / max(np.mean(e0 ** 2), 1e-12))
        out["rel_rmse_pct"] = float(100.0 * out["rmse"] / max(out["baseline_rmse"], 1e-9))
    if std is not None:
        s = np.asarray(std, float)
        out["mean_std"] = float(s.mean())
        out["coverage_2sigma"] = float(np.mean(np.abs(e) <= 2.0 * s + 1e-9))
    return out


def by_time_bin(err, times):
    """Mean |error| in each time interval of TIME_BINS; err has shape (nt, S)."""
    times = np.asarray(times, float)
    out = []
    for lo, hi in TIME_BINS:
        m = (times >= lo) & (times <= hi) if hi >= times.max() else (times >= lo) & (times < hi)
        out.append(dict(interval=[lo, hi], mae=float(np.mean(np.abs(err[m]))) if m.any() else None,
                        max=float(np.max(np.abs(err[m]))) if m.any() else None))
    return out


def by_component(err, tags):
    """Mean |error| per aircraft component; err has shape (nt, S)."""
    out = []
    for i, name in enumerate(geometry.TAGS):
        m = tags == i
        out.append(dict(part=name, n=int(m.sum()),
                        mae=float(np.mean(np.abs(err[:, m]))) if m.any() else None,
                        max=float(np.max(np.abs(err[:, m]))) if m.any() else None))
    return out


def locate_max(err, times, xyz) -> dict:
    nt, s = np.unravel_index(np.argmax(np.abs(err)), err.shape)
    return dict(time=float(times[nt]), xyz=[float(v) for v in xyz[s]], error=float(err[nt, s]), node=int(s))
