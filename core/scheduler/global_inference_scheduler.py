"""
GlobalInferenceScheduler — 全局算力调度。

避免多摄像头同时运行 SCRFD detector 导致 CPU 过载。
简单令牌桶实现：最多 max_concurrent 个 camera 同时推理。
"""
import threading


class GlobalInferenceScheduler:
    """
    全局推理调度器 (thread-safe 令牌桶)。

    用法:
        gs = GlobalInferenceScheduler(max_concurrent=2)
        if gs.acquire("cam0"):
            detector.detect(frame)
            gs.release("cam0")
    """

    def __init__(self, max_concurrent: int = 2):
        self._max = max_concurrent
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._active: set = set()

    def acquire(self, camera_id: str) -> bool:
        """尝试获取推理令牌。非阻塞，失败立即返回 False。"""
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            with self._lock:
                self._active.add(camera_id)
        return acquired

    def release(self, camera_id: str):
        """释放推理令牌。"""
        with self._lock:
            self._active.discard(camera_id)
        try:
            self._semaphore.release()
        except ValueError:
            pass

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def max_concurrent(self) -> int:
        return self._max
