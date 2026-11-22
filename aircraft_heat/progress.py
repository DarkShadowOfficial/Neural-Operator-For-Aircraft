"""Thread-safe progress reporting shown by the demo UI while data are generated / the operator trains."""
from __future__ import annotations

import threading
import time


class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self.stage = "starting"
        self.detail = ""
        self.fraction = 0.0
        self.logs: list[str] = []
        self.t0 = time.time()
        self.stages: list[dict] = []

    def set_stage(self, stage: str, detail: str = ""):
        with self._lock:
            self.stage, self.detail, self.fraction = stage, detail, 0.0
            self.stages.append({"stage": stage, "t": time.time() - self.t0})
            self.logs.append(f"[{time.time() - self.t0:6.1f}s] {stage}: {detail}")

    def update(self, fraction: float, detail: str | None = None):
        with self._lock:
            self.fraction = float(min(max(fraction, 0.0), 1.0))
            if detail is not None:
                self.detail = detail

    def add_log(self, msg: str):
        with self._lock:
            self.logs.append(f"[{time.time() - self.t0:6.1f}s] {msg}")
            self.logs = self.logs[-200:]

    def snapshot(self) -> dict:
        with self._lock:
            return {"stage": self.stage, "detail": self.detail, "fraction": self.fraction,
                    "elapsed_s": time.time() - self.t0, "log": self.logs[-25:]}
