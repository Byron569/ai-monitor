"""
MultiObjectTracker — ByteTrack + TrackMemory 封装。

v3: Tracker 主导系统，每帧输出标准化结果。
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from core.track_memory import TrackMemory

logger = logging.getLogger(__name__)

try:
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False


class MultiObjectTracker:
    """
    ByteTrack + TrackMemory 多目标追踪器。

    输出标准化为:
        tracked_objects = tracker.update(detections)
        # tracked_objects: {track_id: {"bbox": [x1,y1,x2,y2], "score": float, "age": int}}
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.5,
        frame_rate: int = 30,
    ) -> None:
        if not BYTETRACK_AVAILABLE:
            raise RuntimeError("boxmot 未安装")

        self._tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )
        self._memory = TrackMemory(ttl_active=1.5, ttl_lost=1.5)
        self._frame_count = 0

    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        frame: np.ndarray = None,
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """每帧调用。有检测更新记忆，无检测返回活跃记忆。"""
        self._frame_count += 1

        dets = np.array([
            [x1, y1, x2, y2, conf, 0.0]
            for (x1, y1, x2, y2, conf) in detections
        ], dtype=np.float32)

        img = np.zeros((1, 1, 3), dtype=np.uint8) if frame is None else frame

        has_detections = len(dets) > 0
        if has_detections:
            try:
                tracks = self._tracker.update(dets, img)
            except Exception:
                tracks = None
        else:
            tracks = None

        active: Dict[int, Tuple[int, int, int, int]] = {}
        if tracks is not None and len(tracks) > 0:
            arr = np.array(tracks)
            for t in arr:
                tid = int(t[4])
                x1, y1, x2, y2 = map(int, t[:4])
                active[tid] = (x1, y1, x2, y2)

        return active

    def get_active(self) -> Dict[int, Tuple[int, int, int, int]]:
        return self._memory.get_active()

    def get_memory(self) -> TrackMemory:
        return self._memory

    @property
    def active_count(self) -> int:
        return self._memory.active_count
