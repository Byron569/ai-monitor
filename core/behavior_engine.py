"""
BehaviorEngine — 行为状态机。

将轨迹特征转为行为状态:
  MOVING    — 持续移动
  STATIONARY — 长时间静止
  LOITERING — 在区域长时间徘徊
  DISAPPEARED — 目标消失

纯规则，无 AI 模型。
"""

import logging
from enum import Enum
from typing import Dict, Optional

from core.trajectory_analyzer import TrajectoryAnalyzer, TrajectoryFeatures
from core.region_manager import RegionManager

logger = logging.getLogger(__name__)


class Behavior(Enum):
    MOVING = "moving"
    STATIONARY = "stationary"
    LOITERING = "loitering"
    DISAPPEARED = "disappeared"


class BehaviorState:
    __slots__ = ("track_id", "behavior", "confidence", "since_frame",
                 "stationary_start", "prev_change_log")

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.behavior = Behavior.MOVING
        self.confidence = 0.5
        self.since_frame = 0
        self.stationary_start = -1
        self.prev_change_log = ""


class BehaviorEngine:
    """
    行为状态机。

    用法:
        engine = BehaviorEngine(ta, region_mgr, loitering_threshold=300)
        state = engine.update(track_id, trajectory_features)
    """

    def __init__(
        self,
        trajectory_analyzer: TrajectoryAnalyzer,
        region_manager: "RegionManager",
        stationary_threshold: int = 60,
        loitering_threshold: int = 300,
    ):
        self.trajectory_analyzer = trajectory_analyzer
        self.region_manager = region_manager
        self.stationary_threshold = stationary_threshold
        self.loitering_threshold = loitering_threshold
        self._states: Dict[int, BehaviorState] = {}

    def update(self, track_id: int, frame_count: int) -> BehaviorState:
        tf = self.trajectory_analyzer.get(track_id)
        if tf is None:
            return self._ensure_state(track_id)

        if track_id not in self._states:
            self._states[track_id] = BehaviorState(track_id)

        bs = self._states[track_id]

        # 规则判断（优先检查更高阈值，防止 LOITERING 被 STATIONARY 屏蔽）
        if tf.stationary_frames >= self.loitering_threshold:
            new_behavior = Behavior.LOITERING
            confidence = min(1.0, (tf.stationary_frames - self.loitering_threshold) / 100)
        elif tf.stationary_frames >= self.stationary_threshold:
            new_behavior = Behavior.STATIONARY
            confidence = min(1.0, tf.stationary_frames / self.loitering_threshold)
        else:
            new_behavior = Behavior.MOVING
            confidence = tf.movement_score

        if new_behavior != bs.behavior:
            logger.info(
                f"[BEHAVIOR] track={track_id} {bs.behavior.value} → {new_behavior.value}"
            )
            bs.behavior = new_behavior
            bs.since_frame = frame_count

        bs.confidence = confidence
        return bs

    def mark_disappeared(self, track_id: int, frame_count: int) -> BehaviorState:
        bs = self._ensure_state(track_id)
        if bs.behavior != Behavior.DISAPPEARED:
            logger.info(f"[BEHAVIOR] track={track_id} DISAPPEARED")
            bs.behavior = Behavior.DISAPPEARED
            bs.since_frame = frame_count
        return bs

    def get(self, track_id: int) -> Optional[BehaviorState]:
        return self._states.get(track_id)

    def cleanup(self, active_ids: set) -> None:
        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]

    def _ensure_state(self, track_id: int) -> BehaviorState:
        if track_id not in self._states:
            self._states[track_id] = BehaviorState(track_id)
        return self._states[track_id]
