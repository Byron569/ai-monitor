"""
TrajectoryAnalyzer — 轨迹分析器。

从 TrackMemory 提取轨迹特征：速度、方向、静止帧数、移动得分。
纯规则，无 AI 模型。
"""

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryFeatures:
    __slots__ = (
        "track_id", "speed", "direction", "stationary_frames",
        "movement_score", "history",
    )

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.speed = 0.0
        self.direction = 0.0
        self.stationary_frames = 0
        self.movement_score = 0.0
        self.history: deque = deque(maxlen=30)


class TrajectoryAnalyzer:
    """
    轨迹分析器。

    用法:
        ta = TrajectoryAnalyzer(stationary_threshold=60)
        features = ta.analyze(track_id, bbox)
    """

    def __init__(self, stationary_threshold: int = 60):
        self.stationary_threshold = stationary_threshold
        self._tracks: Dict[int, TrajectoryFeatures] = {}

    def analyze(
        self, track_id: int, bbox: Tuple[int, int, int, int]
    ) -> TrajectoryFeatures:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if track_id not in self._tracks:
            tf = TrajectoryFeatures(track_id)
            tf.history.append((cx, cy))
            self._tracks[track_id] = tf
            return tf

        tf = self._tracks[track_id]
        tf.history.append((cx, cy))

        # 速度 = 最近两点之间的像素距离
        if len(tf.history) >= 2:
            h = list(tf.history)
            dx = h[-1][0] - h[-2][0]
            dy = h[-1][1] - h[-2][1]
            tf.speed = (dx ** 2 + dy ** 2) ** 0.5
            tf.direction = np.arctan2(dy, dx) if abs(dx) > 0 or abs(dy) > 0 else 0.0

        # 静止帧数
        if tf.speed < 3.0:
            tf.stationary_frames += 1
        else:
            tf.stationary_frames = max(0, tf.stationary_frames - 1)

        # 移动得分 (0=静止, 1=快速移动)
        tf.movement_score = min(1.0, tf.speed / 20.0)

        return tf

    def get(self, track_id: int) -> Optional[TrajectoryFeatures]:
        return self._tracks.get(track_id)

    def cleanup(self, active_ids: set) -> None:
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]
