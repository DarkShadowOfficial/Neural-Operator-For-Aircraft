# Aircraft aerothermal heating: neural operator vs built-in FEM

Interactive demo (same idea as the 1-D rod demo) that compares a neural operator with a built-in 3-D finite-element
reference solver for a supersonic aircraft. Inputs: flight speed, air temperature, altitude, aircraft material
(k, rho, cp), initial aircraft temperature (+ optional nose-to-tail gradient). Outputs: surface temperature over time
at 11 named aircraft parts, error graphs, 2-D surface maps, speed/complexity comparison, and generalization checks.

## Run
```bash
pip install -r requirements.txt        # numpy + scipy are enough; fastapi/uvicorn optional
python -m demo                         # shipped checkpoint, already adapted -> starts in ~10 s
python -m demo --preset quick --retrain   # regenerate FEM data (~8 min) and retrain (~5 min)
python -m demo --preset full           # 400 FEM runs, finer mesh, bigger ensemble (hours on 1 core)
python -m pytest tests -q              # FEM verification + operator gradient tests
```
Resumable long jobs: `python scripts/make_dataset.py quick --budget 200` and `python scripts/train.py quick --budget 200`
(re-run to continue).

## Start-up adaptation (error-driven fine-tuning)
On launch the demo measures the operator on fresh random scenarios (built-in FEM as oracle) and fine-tunes it where it is
weakest, until the error is below a threshold. Each round: run FEM on `--adapt-probes` new scenarios -> per-case MAE ->
samples per case = 0 if MAE < 2 K, else round(N * MAE / 10 K) (>= 1, capped at 4N): the case itself plus FEM-run neighbours
-> fine-tune the ensemble (25 % of the new samples are held back as validation) -> next round with NEW scenarios.
It stops after **two consecutive fresh sets** with mean MAE <= `--adapt-target` (10 K) and worst case <= 2x that, or when
`--adapt-budget` seconds are used up. Every round is checkpointed (`checkpoints/<preset>/adapted/`), so it resumes and,
once the target is reached, later launches skip it. Flags: `--no-adapt`, `--adapt-target`, `--adapt-worst`,
`--adapt-budget`, `--adapt-probes`, `--adapt-n`. Offline/resumable: `python scripts/adapt.py quick --budget 90`.

Measured on the shipped quick checkpoint (5 rounds, 79 adaptive FEM samples, ~10 min of FEM + fine-tuning):
untouched 16-scenario test set 13.4 K -> 11.2 K MAE (14.3 % -> 12.0 % of baseline error, max error 473 -> 377 K).
The 8-10-scenario probe sets reached the 10 K target, but they are noisy (case errors span ~2-50 K); the fixed test set
still averages above 10 K because it contains a few hard cases. More rounds/samples keep helping (data-limited).

## What is inside
| Module | Role |
|---|---|
| `aircraft_heat/geometry.py` | implicit aircraft (fuselage, delta wing, fin, pod) in the 20 x 13 x 5.2 m CAD box |
| `aircraft_heat/mesh.py` | layered skin mesh: surface triangulation extruded inward in graded prisms -> tetrahedra |
| `aircraft_heat/aero.py` | stand-in for CFD: local h and recovery temperature (Newtonian edge state, Eckert reference temperature, Sutton-Graves stagnation) |
| `aircraft_heat/fem.py` | P1 tetra FEM, TR-BDF2, one sparse LU per run |
| `aircraft_heat/operator.py` | DeepONet ensemble in NumPy (hand-written backprop) |
| `aircraft_heat/pipeline.py` | FEM data -> training -> held-out evaluation -> learning curve -> scaling benchmark |
| `aircraft_heat/adapt.py` | start-up error-driven fine-tuning (active learning with the FEM as oracle) |
| `demo/` | web UI + FastAPI / stdlib servers |

## Verification (see `tests/`)
Plate vs exact erfc solution <= 1.4 % of driving dT (0.03 % at 60 s); lumped capacitance 0.08 %; energy balance 0.008 %;
smooth-panel mesh convergence within ~10 K (thin tips/edges are mesh-limited, 50-150 K coarse vs medium).

## Shipped quick checkpoint: held-out results (16 unseen FEM scenarios)
Base model: test MAE 13.4 K (train 9.4 K), 14 % of the "nothing happens" error, skill 0.98. Learning curve 26 -> 18 -> 13.4 K for
25 -> 50 -> 100 runs (still falling: data-limited). Worst case 57 K MAE. Ensemble 2-sigma coverage 67 % (over-confident).
FEM ~4 s vs operator 1-4 ms (~1000x) on the coarse mesh.

## Limitations (read before trusting numbers)
* **The FEM is not calibrated to SimScale.** It uses textbook boundary-layer correlations, not CFD. Calibrate `h_scale`
  (`aircraft_heat/aero.py`) against your SimScale surface temperatures before reading operator accuracy as SimScale accuracy.
* Skin-layer FEM (columns <= 0.45 m, adiabatic bottom), no radiation, zero angle of attack, ideal gas, Mach 0.2-4.5.
  ~4 % of surface area is dropped where prisms are degenerate.
* Operator predicts surface temperature only, and uses area-mean h / T_aw from the aero model as input features.
* NumPy implementation (no PyTorch). The FastAPI server was not run in the build sandbox; the stdlib server (same routes) was.
