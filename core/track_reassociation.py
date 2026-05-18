"""
TrackReassociation — 轨迹重关联引擎。

当 track 丢失后，用 SCRFD 检测结果与 TrackMemory 进行匹配，
恢复 track_id。

匹配策略（轻量，不引入重 ReID）:
    1. IoU 匹配 — 直接 bbox 重叠
    2. 空间邻近 — 距离优先
    3. 速度一致性 — 预测位置匹配

日志:
    [REASSOC] track=5 re-linked via IoU (iou=0.85)
    [REASSOC] track=3 re-linked via spatial proximity (dist=12px)
    [REASSOC] no match for detection
"""

import logging
from typing import Dict, List, Optional, Tuple

from core.track_memory import TrackMemory, TrackState

logger = logging.getLogger(__name__)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ca = _center(a)
    cb = _center(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


class TrackReassociation:
    """
    轨迹重关联引擎。

    参数:
        iou_threshold: IoU 匹配阈值
        distance_threshold: 空间邻近匹配的最大像素距离

    用法:
        reassoc = TrackReassociation()
        matches = reassoc.match(detections, track_memory)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        distance_threshold: float = 100,
    ):
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold

    def match(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        track_memory: TrackMemory,
    ) -> Dict[int, Tuple[int, int, int, int, float]]:
        """
        用 SCRFD 检测结果匹配 TrackMemory 中的 lost track。

        Args:
            detections: [(x1,y1,x2,y2,conf), ...]
            track_memory: TrackMemory 实例

        Returns:
            已匹配的 detections 字典:
                {det_idx: (x1,y1,x2,y2,conf, matched_track_id)}
            未匹配的 detections 为新人。
        """
        lost_tracks = track_memory.get_lost()
        if not lost_tracks or not detections:
            return {}

        matched_dets = {}
        used_tids = set()

        # --- 1. IoU 匹配 ---
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            bbox_det = det[:4]
            best_iou, best_tid = 0, None
            for tid, ts in lost_tracks.items():
                if tid in used_tids:
                    continue
                iou_val = _iou(bbox_det, ts.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid
            if best_iou >= self.iou_threshold and best_tid is not None:
                matched_dets[i] = (*det, best_tid)
                used_tids.add(best_tid)
                logger.info(f"[REASSOC] track={best_tid} re-linked via IoU (iou={best_iou:.2f})")

        # --- 2. 空间邻近 ---
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            bbox_det = det[:4]
            best_dist, best_tid = float("inf"), None
            for tid, ts in lost_tracks.items():
                if tid in used_tids:
                    continue
                dist = _distance(bbox_det, ts.bbox)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            if best_dist <= self.distance_threshold and best_tid is not None:
                matched_dets[i] = (*det, best_tid)
                used_tids.add(best_tid)
                logger.info(f"[REASSOC] track={best_tid} re-linked via spatial (dist={best_dist:.0f}px)")

        return matched_dets
