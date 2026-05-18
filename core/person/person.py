"""
Person 对象 — 以"人"为中心的抽象单元。

后续扩展：
  - Person.pose_state: "standing" / "sitting" / "falling"
  - Person.is_fall: 跌倒标志
  - Person.behavior_history: 行为序列
  - Person → Event Engine
"""

import time
from typing import Optional

import numpy as np


class Person:
    """
    单个人体追踪对象。

    属性:
        track_id:       全局唯一追踪 ID
        identity:       识别身份（"Byron" / "Unknown"）
        bbox:           当前边界框 (x1, y1, x2, y2)
        confidence:     检测置信度
        embedding:      ArcFace 512-d 向量（已归一化）
        last_seen:      最后出现时间戳
        pose_state:     姿态状态（预留）"standing" / "sitting" / "falling"
        is_fall:        是否跌倒（预留）
    """

    __slots__ = (
        "track_id", "identity", "bbox", "confidence",
        "embedding", "last_seen", "pose_state", "is_fall",
        "frame_seen",
    )

    def __init__(
        self,
        track_id: int,
        bbox: tuple,
        confidence: float = 0.0,
    ) -> None:
        self.track_id = track_id
        self.identity = "Unknown"
        self.bbox = bbox
        self.confidence = confidence
        self.embedding: Optional[np.ndarray] = None
        self.last_seen = time.time()
        self.pose_state = "unknown"
        self.is_fall = False
        self.frame_seen = 1

    @property
    def is_identified(self) -> bool:
        return self.identity != "Unknown"

    def mark_identified(self, name: str, embedding: np.ndarray) -> None:
        self.identity = name
        self.embedding = embedding
        self.last_seen = time.time()

    def update_bbox(self, bbox: tuple, confidence: float) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frame_seen += 1

    def __repr__(self) -> str:
        ident = self.identity if self.is_identified else "?"
        return f"Person(id={self.track_id}, name={ident})"
