"""
VisionTask — 可插拔视觉任务接口。

为后续融合 YOLOv8-Pose 摔倒检测等扩展任务预留统一接口。
所有扩展任务必须继承此基类，通过 Pipeline.tasks 列表接入。
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VisionEvent:
    """统一事件数据结构。"""

    __slots__ = ("event_type", "track_id", "confidence", "timestamp", "payload")

    def __init__(
        self,
        event_type: str,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        payload: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.track_id = track_id
        self.confidence = confidence
        self.timestamp = time.time()
        self.payload = payload or {}

    def __repr__(self):
        return (
            f"VisionEvent(type={self.event_type}, tid={self.track_id}, "
            f"conf={self.confidence:.2f})"
        )


class VisionTask(ABC):
    """
    可插拔视觉任务基类。

    用法:
        class FallDetectionTask(VisionTask):
            name = "fall_detection"
            enabled = False
            interval = 5

            def should_run(self, frame_id, tracks, context):
                return frame_id % self.interval == 0 and len(tracks) > 0

            def run(self, frame, tracks, context):
                return []  # stub: no events

    接入 Pipeline:
        tasks = [FallDetectionTask()]
        Pipeline(..., tasks=tasks)
    """

    def __init__(self):
        self.name: str = self.__class__.__name__
        self.enabled: bool = False
        self.interval: int = 5

    @abstractmethod
    def should_run(self, frame_id: int, tracks: list, context: dict) -> bool:
        """
        判断当前帧是否需要运行此任务。

        Args:
            frame_id: 当前帧序号
            tracks: 活跃 track 列表 [(track_id, bbox, identity), ...]
            context: 共享上下文 (person_manager, event_system, etc.)
        """
        ...

    @abstractmethod
    def run(self, frame, tracks: list, context: dict) -> List[VisionEvent]:
        """
        执行视觉任务。

        Args:
            frame: 当前帧 (numpy array, BGR)
            tracks: 活跃 track 列表
            context: 共享上下文 dict

        Returns:
            List[VisionEvent]: 任务产出的事件
        """
        ...
