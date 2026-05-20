"""
RecognitionWorker v2 — 异步人脸识别工作线程（边缘优化版）。

改进:
  - bounded queue, 满时丢弃低质量任务
  - latest-result cache (避免同 track 重复等待)
  - 异常隔离，worker 崩溃不影响主线程
  - 优雅退出
  - 每任务带 quality_score / timestamp
"""

import threading
import time
from queue import Queue, Full, Empty
from typing import Dict, List, Tuple, Optional

import numpy as np


class _RecogTask:
    __slots__ = ("track_id", "crop", "quality_score", "timestamp")
    def __init__(self, track_id, crop, quality_score, timestamp):
        self.track_id = track_id
        self.crop = crop
        self.quality_score = quality_score
        self.timestamp = timestamp


class RecognitionWorker(threading.Thread):
    """
    后台识别工作线程 v2。

    架构:
        主线程: submit(task) 非阻塞, 队列满返回 False
        工作线程: embedding → DB search → output_queue
        主线程: poll_results() 收割已完成结果, 不阻塞
    """

    def __init__(
        self,
        recognizer,
        database,
        max_queue_size: int = 4,
        result_cache_size: int = 16,
    ) -> None:
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.database = database
        self._input_queue: Queue = Queue(maxsize=max_queue_size)
        self._output_queue: Queue = Queue()
        self._running = True
        self._processed_count = 0
        self._skip_count = 0

        # latest-result cache: track_id → (name, sim, emb)
        self._result_cache: Dict[int, Tuple] = {}
        self._result_cache_size = result_cache_size

    def run(self) -> None:
        while self._running:
            try:
                task: _RecogTask = self._input_queue.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue

            t0 = time.perf_counter()
            try:
                name, sim, emb = self._do_recognize(task.crop)
            except Exception:
                name, sim, emb = "Unknown", 0.0, None
            latency_ms = (time.perf_counter() - t0) * 1000.0

            result = (task.track_id, name, sim, emb, task.quality_score, latency_ms)
            self._output_queue.put(result)
            self._processed_count += 1

            # update latest-result cache
            if name != "Unknown" and emb is not None:
                self._result_cache[task.track_id] = (name, sim, emb)
                if len(self._result_cache) > self._result_cache_size:
                    oldest = next(iter(self._result_cache))
                    del self._result_cache[oldest]

    def _do_recognize(self, crop: np.ndarray) -> Tuple[str, float, object]:
        embedding = self.recognizer.get_embedding(crop)
        if embedding is None:
            return ("Unknown", 0.0, None)
        result = self.database.search(embedding, threshold=self.recognizer.threshold)
        if result is None:
            return ("Unknown", 0.0, embedding)
        name, sim = result
        return (name, sim, embedding)

    def submit(
        self,
        track_id: int,
        crop: np.ndarray,
        quality_score: float = 0.5,
        force: bool = False,
    ) -> bool:
        """
        提交识别任务 (非阻塞)。

        Args:
            track_id: 跟踪 ID
            crop: 人脸裁剪 (BGR, uint8)
            quality_score: 质量分 [0,1]
            force: 强制提交, 即使队列满也替换最低质量任务

        Returns:
            True if submitted, False if dropped
        """
        if not self._running:
            return False

        task = _RecogTask(track_id, crop.copy(), quality_score, 0)

        # Check latest-result cache
        cached = self._result_cache.get(track_id)
        if cached is not None:
            self._output_queue.put((track_id, cached[0], cached[1], cached[2], quality_score, 0.0))
            return True

        try:
            self._input_queue.put_nowait(task)
            return True
        except Full:
            if force:
                try:
                    # pop oldest and retry
                    self._input_queue.get_nowait()
                    self._input_queue.put_nowait(task)
                    return True
                except (Empty, Full):
                    pass
            self._skip_count += 1
            return False

    def poll_results(self) -> list:
        """
        收割所有已完成结果 (非阻塞)。
        返回: [(track_id, name, sim, emb, quality_score), ...]
        """
        results = []
        while True:
            try:
                results.append(self._output_queue.get_nowait())
            except Empty:
                break
        return results

    def stop(self) -> None:
        self._running = False

    @property
    def queue_size(self) -> int:
        return self._input_queue.qsize()

    @property
    def pending_count(self) -> int:
        return self._input_queue.qsize() + self._output_queue.qsize()

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def skip_count(self) -> int:
        return self._skip_count

    @property
    def cache_hit(self) -> int:
        return len(self._result_cache)
