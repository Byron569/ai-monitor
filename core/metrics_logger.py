"""
MetricsLogger — 统一指标记录器
"""

import time
from collections import deque
from typing import Dict


class MetricsLogger:
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histories: Dict[str, deque] = {}
        self._start_time = time.time()

    def inc(self, name: str, delta: int = 1):
        self._counters[name] = self._counters.get(name, 0) + delta

    def set(self, name: str, value: float):
        self._gauges[name] = value

    def get(self, name: str, default=0):
        return self._counters.get(name, self._gauges.get(name, default))

    def elapsed(self) -> float:
        return time.time() - self._start_time

    def summary(self) -> str:
        lines = ["[METRICS]"]
        lines.append(f"  runtime={self.elapsed():.1f}s")
        for k, v in self._counters.items():
            lines.append(f"  {k}={v}")
        for k, v in self._gauges.items():
            lines.append(f"  {k}={v:.2f}")
        return "\n".join(lines)
