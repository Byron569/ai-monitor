"""
GlobalMetrics — 多摄像头全局统计。
"""
import time
import threading


class GlobalMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._camera_metrics = {}
        self._start_time = time.time()
        self.total_recognition_enqueued = 0
        self.total_recognition_skipped = 0

    def register(self, camera_id: str, metrics):
        with self._lock:
            self._camera_metrics[camera_id] = metrics

    def unregister(self, camera_id: str):
        with self._lock:
            self._camera_metrics.pop(camera_id, None)

    @property
    def total_fps(self) -> float:
        with self._lock:
            if not self._camera_metrics:
                return 0.0
            return sum(m.fps for m in self._camera_metrics.values())

    @property
    def total_cameras(self) -> int:
        with self._lock:
            return len(self._camera_metrics)

    @property
    def avg_detection_ms(self) -> float:
        with self._lock:
            valid = [m.avg_detection_ms for m in self._camera_metrics.values() if m.avg_detection_ms > 0]
            return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_recognition_ms(self) -> float:
        with self._lock:
            valid = [m.avg_recognition_ms for m in self._camera_metrics.values() if m.avg_recognition_ms > 0]
            return sum(valid) / len(valid) if valid else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def summary(self) -> str:
        lines = ["=" * 55, " MULTI-CAMERA REPORT", "=" * 55]
        with self._lock:
            lines.append(f"  Cameras: {self.total_cameras} | Total FPS: {self.total_fps:.1f} | Runtime: {self.elapsed:.0f}s")
            lines.append(f"  Avg Detect: {self.avg_detection_ms:.1f}ms | Avg Recog: {self.avg_recognition_ms:.1f}ms")
            for cid, m in self._camera_metrics.items():
                s = m.summary()
                lines.append(f"  [{cid}] fps={s['fps']} det={s['detect_ms']}ms trk={s['track_ms']}ms rec={s['recog_ms']}ms frames={s['frames']}")
        return "\n".join(lines)
