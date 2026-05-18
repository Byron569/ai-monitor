"""
CameraMetrics — per-camera 性能统计。
"""
import time
import threading
from collections import deque


class CameraMetrics:
    def __init__(self, camera_id: str, window_size: int = 30):
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._window_size = window_size

        self._detect_times = deque(maxlen=window_size)
        self._track_times = deque(maxlen=window_size)
        self._recognize_times = deque(maxlen=window_size)

        self._fps_times = deque(maxlen=window_size)
        self._last_frame_time = time.time()

        self.reconnect_count = 0
        self.dropped_frames = 0
        self.total_frames = 0

    def record_frame(self):
        now = time.time()
        with self._lock:
            dt = now - self._last_frame_time
            if dt > 0:
                self._fps_times.append(1.0 / dt)
            self._last_frame_time = now
            self.total_frames += 1

    def record_detect(self, ms: float):
        with self._lock:
            self._detect_times.append(ms)

    def record_track(self, ms: float):
        with self._lock:
            self._track_times.append(ms)

    def record_recognize(self, ms: float):
        with self._lock:
            self._recognize_times.append(ms)

    def record_drop(self):
        with self._lock:
            self.dropped_frames += 1

    def record_reconnect(self):
        with self._lock:
            self.reconnect_count += 1

    @property
    def fps(self) -> float:
        with self._lock:
            if not self._fps_times:
                return 0.0
            return sum(self._fps_times) / len(self._fps_times)

    @property
    def avg_detection_ms(self) -> float:
        with self._lock:
            if not self._detect_times:
                return 0.0
            return sum(self._detect_times) / len(self._detect_times)

    @property
    def avg_tracking_ms(self) -> float:
        with self._lock:
            if not self._track_times:
                return 0.0
            return sum(self._track_times) / len(self._track_times)

    @property
    def avg_recognition_ms(self) -> float:
        with self._lock:
            if not self._recognize_times:
                return 0.0
            return sum(self._recognize_times) / len(self._recognize_times)

    def summary(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "fps": round(self.fps, 1),
            "detect_ms": round(self.avg_detection_ms, 1),
            "track_ms": round(self.avg_tracking_ms, 1),
            "recog_ms": round(self.avg_recognition_ms, 1),
            "frames": self.total_frames,
            "dropped": self.dropped_frames,
            "reconnects": self.reconnect_count,
        }
