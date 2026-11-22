"""Train the operator for a preset from the cached FEM dataset (resumable):

    python scripts/train.py quick [--budget 200]

Finished ensemble members and learning-curve points are cached, so after an interruption (or --budget) simply run again.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aircraft_heat.config import get_preset                          # noqa: E402
from aircraft_heat.persistence import save_bundle                      # noqa: E402
from aircraft_heat.pipeline import BudgetExceeded, run_pipeline        # noqa: E402
from aircraft_heat.progress import Progress                            # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="quick")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--budget", type=float, default=None, help="stop after this many seconds (resume by re-running)")
    a = ap.parse_args()
    cfg, ck = get_preset(a.preset), Path(a.ckpt_dir) / a.preset
    prog = Progress()
    t0 = time.time()
    try:
        b = run_pipeline(cfg, prog, ck, budget_s=a.budget)
    except BudgetExceeded:
        print("\n".join(prog.snapshot()["log"][-6:]))
        print("budget reached; run again to continue")
        sys.exit(0)
    save_bundle(b, ck)
    print("\n".join(prog.snapshot()["log"][-8:]))
    e = b["meta"]["eval"]
    print(f"done in {time.time() - t0:.0f} s | test MAE {e['test']['mae']:.2f} K, rel {e['test']['rel_rmse_pct']:.1f} %, "
          f"skill {e['test']['skill']:.3f} | train MAE {e['train']['mae']:.2f} K -> {ck}")
