"""
性能监控器 v2 — 实时统计 Pipeline 各阶段耗时与识别指标。
"""

import time
import threading
from collections import deque
from typing import Dict


class PerformanceMonitor:
    """
    轻量级性能统计器 v2。

    支持: 阶段耗时 + 识别计数器
    """

    def __init__(self, window_size: int = 30) -> None:
        self._tracks: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._active_ticks: Dict[str, float] = {}
        self._window_size = window_size
        self._lock = threading.Lock()
        self._report_count = 0

    def tick(self, label: str) -> None:
        with self._lock:
            self._active_ticks[label] = time.perf_counter()

    def tock(self, start_label: str, track_label: str) -> float:
        now = time.perf_counter()
        with self._lock:
            t0 = self._active_ticks.pop(start_label, now)
            elapsed = (now - t0) * 1000
            if track_label not in self._tracks:
                self._tracks[track_label] = deque(maxlen=self._window_size)
            self._tracks[track_label].append(elapsed)
            return elapsed

    # --- Recognition counters ---
    def inc(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def get(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    # --- Report ---
    def should_report(self, frame_id: int, interval: int = 60) -> bool:
        return frame_id > 0 and frame_id % interval == 0

    def report(self) -> str:
        with self._lock:
            parts = []
            for label, dq in self._tracks.items():
                if dq:
                    avg = sum(dq) / len(dq)
                    parts.append(f"{label}:{avg:4.0f}ms")
            return " | ".join(parts) if parts else "no data"

    def recog_report(self) -> str:
        with self._lock:
            enq = self._counters.get("recog_enqueue", 0)
            skip = self._counters.get("recog_skip", 0)
            reject = self._counters.get("recog_quality_reject", 0)
            cache_hit = self._counters.get("recog_cache_hit", 0)
            queue = self._gauges.get("recog_queue_size", 0)
            worker_ms = 0.0
            if "recognize" in self._tracks and self._tracks["recognize"]:
                worker_ms = sum(self._tracks["recognize"]) / len(self._tracks["recognize"])
            return (
                f"[RECOG] enqueue={enq} skip={skip} reject={reject} "
                f"cache_hit={cache_hit} queue={int(queue)} worker={worker_ms:.1f}ms"
            )
