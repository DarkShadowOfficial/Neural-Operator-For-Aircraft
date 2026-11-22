"""Generate (and cache) the FEM dataset of a preset, resumable:

    python scripts/make_dataset.py quick [--budget 600]

Runs are saved in chunks, so the script can be interrupted and restarted; with --budget it stops after that
many seconds (re-run to continue).  The final file checkpoints/<preset>/dataset.npz is what training reads.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aircraft_heat import config                                   # noqa: E402
from aircraft_heat.config import MAX_TIME, get_preset                # noqa: E402
from aircraft_heat.data import run_fem_batch                         # noqa: E402

CHUNK = 8

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="quick")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--budget", type=float, default=1e9)
    a = ap.parse_args()
    cfg = get_preset(a.preset)
    out_dir = Path(a.ckpt_dir) / a.preset
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, MAX_TIME, cfg.n_times)
    t0 = time.time()
    done_all = True
    final = dict(times=times, key=np.array([cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_times, cfg.seed]), mesh=np.array(cfg.mesh))
    for name, n, seed in (("train", cfg.n_train, cfg.seed), ("val", cfg.n_val, cfg.seed + 101), ("test", cfg.n_test, cfg.seed + 202)):
        vecs = config.sample_conditions(n, np.random.default_rng(seed))
        Ts, ws = [], []
        for lo in range(0, n, CHUNK):
            f = parts / f"{name}_{lo:04d}.npz"
            if not f.exists():
                if time.time() - t0 > a.budget:
                    done_all = False
                    break
                T, w = run_fem_batch(vecs[lo:lo + CHUNK], times, cfg.mesh)
                np.savez(f, T=T, w=w)
                print(f"{name}: {min(lo + CHUNK, n)}/{n}  ({time.time() - t0:.0f} s)", flush=True)
            d = np.load(f)
            Ts.append(d["T"]); ws.append(d["w"])
        if not done_all:
            break
        final[f"{name}_c"], final[f"{name}_T"], final[f"{name}_wall"] = vecs, np.concatenate(Ts), np.concatenate(ws)
    if done_all:
        np.savez_compressed(out_dir / "dataset.npz", **final)
        print(f"dataset complete -> {out_dir / 'dataset.npz'}  (mean FEM run {np.concatenate([final['train_wall'], final['val_wall'], final['test_wall']]).mean():.2f} s)")
    else:
        print("budget reached; run again to continue")
