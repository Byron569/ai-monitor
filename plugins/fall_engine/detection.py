"""
标准检测数据结构
所有 backend 必须返回此格式的对象，不允许返回 dict 或 numpy 类型
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Keypoint:
    """单个关键点"""
    x: float
    y: float
    confidence: float

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.confidence = float(self.confidence)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict) -> "Keypoint":
        return cls(x=float(d["x"]), y=float(d["y"]), confidence=float(d["confidence"]))

    def to_list(self) -> list:
        return [self.x, self.y, self.confidence]


@dataclass
class Detection:
    """统一检测结果"""
    bbox: list           # [x1, y1, x2, y2] 均为 float
    score: float
    class_id: int = 0
    keypoints: list = field(default_factory=list)  # list[Keypoint]
    track_id: Optional[int] = None
    label: str = "person"

    def __post_init__(self):
        self.bbox = [float(v) for v in self.bbox]
        self.score = float(self.score)
        self.class_id = int(self.class_id)

    def to_dict(self) -> dict:
        return {
            "detection_id": id(self),
            "bbox": self.bbox,
            "score": self.score,
            "class_id": self.class_id,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "track_id": self.track_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        kps = [Keypoint.from_dict(k) for k in d.get("keypoints", [])]
        return cls(
            bbox=d["bbox"],
            score=d["score"],
            class_id=d.get("class_id", 0),
            keypoints=kps,
            track_id=d.get("track_id"),
            label=d.get("label", "person"),
        )
