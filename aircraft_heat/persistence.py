"""Save / load a trained operator bundle (weights in .npz, everything else as JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Preset

VERSION = 3


def save_bundle(bundle: dict, ckpt_dir) -> None:
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for i, (cfg, st) in enumerate(zip(bundle["model_cfgs"], bundle["model_states"])):
        for k, v in st.items():
            arrs[f"m{i}__{k}"] = v
    np.savez_compressed(d / "models.npz", **arrs)
    meta = dict(bundle["meta"], model_cfgs=bundle["model_cfgs"], version=VERSION)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def load_bundle(ckpt_dir, cfg: Preset):
    d = Path(ckpt_dir)
    if not (d / "models.npz").exists() or not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("version") != VERSION or meta.get("preset") != cfg.name or meta.get("mesh") != cfg.mesh:
        return None
    z = np.load(d / "models.npz")
    states = []
    for i in range(len(meta["model_cfgs"])):
        pre = f"m{i}__"
        states.append({k[len(pre):]: z[k] for k in z.files if k.startswith(pre)})
    return dict(model_cfgs=meta["model_cfgs"], model_states=states, meta=meta)
