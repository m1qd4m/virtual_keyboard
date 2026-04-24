"""fps_counter.py — Rolling-window FPS counter."""

import time
from collections import deque


class FPSCounter:
    def __init__(self, smoothing: int = 20):
        self._times = deque(maxlen=smoothing)
        self._last  = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    def get(self) -> float:
        if len(self._times) < 2:
            return 0.0
        mean_dt = sum(self._times) / len(self._times)
        return 1.0 / mean_dt if mean_dt > 0 else 0.0
