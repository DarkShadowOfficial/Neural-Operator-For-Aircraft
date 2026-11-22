"""FastAPI backend for the interactive demo (same structure as the 1-D rod demo).

``/api/status`` shows training progress; ``/api/simulate``, ``/api/snapshot`` and ``/api/stress`` compare the
operator with the built-in FEM for any conditions.  The logic lives in ``demo.core`` so it can also be served
by ``demo.stdlib_server`` when FastAPI is not installed.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo.core import DemoApp

STATIC = Path(__file__).parent / "static"


class SimRequest(BaseModel):
    velocity: float = 1000.0
    T_air: float = 400.0
    altitude: float = 0.0
    T_init: float = 350.0
    dT_axial: float = 0.0
    k: float = 6.7
    rho: float = 4430.0
    cp: float = 526.0
    material: str | None = None
    t: float = 60.0
    n_times: int = 61
    n_cases: int = 6
    seed: int | None = None


def create_app(preset: str = "quick", ckpt_root: str = "checkpoints", retrain: bool = False,
               adapt: dict | None = None) -> FastAPI:
    core = DemoApp(preset, ckpt_root, retrain, adapt)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        core.start()
        yield

    app = FastAPI(title="Aircraft heat operator demo", lifespan=lifespan)
    app.state.core = core
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def call(fn, *a):
        try:
            return fn(*a)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return core.status()

    @app.get("/api/config")
    def config():
        return core.config()

    @app.get("/api/random_case")
    def random_case(seed: int | None = None):
        return core.random_case(seed)

    @app.post("/api/simulate")
    def simulate(req: SimRequest):
        return call(core.simulate, req.model_dump())

    @app.post("/api/snapshot")
    def snapshot(req: SimRequest):
        return call(core.snapshot, req.model_dump())

    @app.post("/api/stress")
    def stress(req: SimRequest):
        return call(core.stress, req.model_dump())

    return app


app = create_app(os.environ.get("AIRCRAFT_PRESET", "quick"), os.environ.get("AIRCRAFT_CKPT_DIR", "checkpoints"),
                 os.environ.get("AIRCRAFT_RETRAIN", "0") == "1") if os.environ.get("AIRCRAFT_AUTOCREATE") else None
