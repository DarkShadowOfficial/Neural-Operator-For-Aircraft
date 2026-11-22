"""Run the error-driven adaptation offline (resumable; each round is checkpointed to checkpoints/<preset>/adapted):

    python scripts/adapt.py quick --target 10 --budget 200      # re-run to continue
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aircraft_heat.config import get_preset                                    # noqa: E402
from aircraft_heat.persistence import load_bundle                                # noqa: E402
from aircraft_heat.pipeline import load_adapted, run_adaptation                  # noqa: E402
from aircraft_heat.progress import Progress                                      # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="quick")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--target", type=float, default=10.0)
    ap.add_argument("--worst", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("-N", type=int, default=3)
    a = ap.parse_args()
    cfg, ck = get_preset(a.preset), Path(a.ckpt_dir) / a.preset
    bundle = load_adapted(ck, cfg) or load_bundle(ck, cfg)
    prog = Progress()
    t0 = time.time()
    b = run_adaptation(bundle, cfg, prog, ck, target=a.target, worst_factor=a.worst, budget_s=a.budget, n_probe=a.probes, N=a.N)
    print("\n".join(l for l in prog.snapshot()["log"] if "adapt" in l))
    ad = b["meta"]["adaptation"]
    print(f"{time.time() - t0:.0f} s | reached={ad['reached']} ({ad['stop_reason']}) | rounds={len(ad['rounds'])} | adaptive samples={ad['n_extra']}")
    print("held-out test set  before:", {k: round(v, 3) for k, v in ad["test_before"].items() if k in ("mae", "rel_rmse_pct", "skill", "max")})
    print("held-out test set  after: ", {k: round(v, 3) for k, v in ad["test_after"].items() if k in ("mae", "rel_rmse_pct", "skill", "max")})
