"""
目标追踪模块 — 基于 IoU + 外观 embedding 的轻量级 TrackManager。

在不引入 ByteTrack 依赖的情况下，提供 Track ID 分配与身份缓存能力。
后续可无缝替换为 ByteTrack / DeepSORT。
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


class Track:
    """单条跟踪轨迹。"""

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.bbox = bbox
        self.disappeared = 0
        self.name: str = ""
        self.embedding: Optional[np.ndarray] = None
        self.last_seen: float = time.time()

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        self.bbox = bbox
        self.disappeared = 0
        self.last_seen = time.time()

    def mark_missed(self) -> None:
        self.disappeared += 1


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """计算两个 bbox 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class TrackManager:
    """
    轻量级 IoU Tracker + Identity Cache。

    职责：
      1. 为每个检测分配稳定 Track ID
      2. 已识别的 Track 缓存其 identity，不再重复调用 embedding
      3. Track 消失 N 帧后自动清除

    接口与 ByteTrack 对齐，后续可无缝替换。
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        iou_threshold: float = 0.3,
        max_tracks: int = 50,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_tracks = max_tracks
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0
        self._frame_count = 0

    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """
        更新跟踪状态。

        Args:
            detections: YOLO 检测结果 [(x1,y1,x2,y2,conf), ...]。

        Returns:
            Dict[track_id → bbox]: 当前活跃的 Track ID 及其边界框。
        """
        self._frame_count += 1

        if len(detections) == 0:
            for t in list(self._tracks.values()):
                t.mark_missed()
                if t.disappeared > self.max_disappeared:
                    del self._tracks[t.track_id]
            return {}

        # 所有 track 标记为未匹配
        for t in self._tracks.values():
            t.mark_missed()

        # 构建 IoU 矩阵
        active_tracks = [t for t in self._tracks.values() if t.track_id in self._tracks]
        assigned_dets = set()
        assigned_tracks = set()

        for t in active_tracks:
            if len(detections) == 0:
                break
            best_iou = 0.0
            best_idx = -1
            for i, (x1, y1, x2, y2, _conf) in enumerate(detections):
                if i in assigned_dets:
                    continue
                score = iou(t.bbox, (x1, y1, x2, y2))
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_iou >= self.iou_threshold:
                t.update((detections[best_idx][0], detections[best_idx][1],
                          detections[best_idx][2], detections[best_idx][3]))
                assigned_dets.add(best_idx)
                assigned_tracks.add(t.track_id)

        # 为新检测创建 track
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue
            if len(self._tracks) >= self.max_tracks:
                break
            x1, y1, x2, y2, _conf = det
            track = Track(self._next_id, (x1, y1, x2, y2))
            self._tracks[self._next_id] = track
            self._next_id += 1

        # 清理过期 track
        expired = [tid for tid, t in self._tracks.items()
                   if t.disappeared > self.max_disappeared]
        for tid in expired:
            del self._tracks[tid]

    def get_active(self) -> Dict[int, Tuple[int, int, int, int]]:
        """返回所有活跃 track 的 (track_id → bbox) 映射。"""
        return {tid: t.bbox for tid, t in self._tracks.items()
                if t.disappeared <= self.max_disappeared}

    def get_identity(self, track_id: int) -> Optional[str]:
        """获取 track 的缓存身份。"""
        t = self._tracks.get(track_id)
        return t.name if t and t.name else None

    def set_identity(
        self, track_id: int, name: str, embedding: np.ndarray
    ) -> None:
        """缓存 track 的识别结果。"""
        t = self._tracks.get(track_id)
        if t:
            t.name = name
            t.embedding = embedding

    def is_identified(self, track_id: int) -> bool:
        """track 是否已被识别。"""
        t = self._tracks.get(track_id)
        return t is not None and bool(t.name)

    @property
    def active_count(self) -> int:
        return len([t for t in self._tracks.values()
                    if t.disappeared <= self.max_disappeared])
