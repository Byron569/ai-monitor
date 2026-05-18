"""
LightweightIoUTracker — 轻量 IoU 跟踪器 (零 boxmot 依赖)。

纯 Python + numpy，适合边缘 CPU。
兼容当前 Pipeline 的 TrackMemory 接口。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.track_memory import TrackMemory


def _calc_iou(a: Tuple, b: Tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


class _IoUTrack:
    """内部轻量 track 状态。"""
    __slots__ = ("track_id", "bbox", "score", "hits", "age", "lost", "confirmed", "created_at")

    def __init__(self, tid: int, bbox: Tuple, score: float):
        self.track_id = tid
        self.bbox = bbox
        self.score = score
        self.hits = 1
        self.age = 1
        self.lost = 0
        self.confirmed = False
        self.created_at = time.time()


class LightweightIoUTracker:
    """
    轻量 IoU 跟踪器。

    参数:
        iou_threshold: IoU 匹配阈值 (默认 0.3)
        max_lost:      最大丢失帧数 (默认 15)
        min_hits:      确认所需最少命中帧数 (默认 2)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 15,
        min_hits: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self._tracks: Dict[int, _IoUTrack] = {}
        self._next_tid = 1

        # TrackMemory 兼容层：提供 Pipeline 需要的接口
        self._memory = TrackMemory(ttl_active=1.5, ttl_lost=max_lost / 30.0)

    def update(
        self,
        detections: List[Any],
        frame=None,
    ) -> Dict[int, Tuple]:
        if not detections:
            # 无检测：所有 track lost++
            for tid in list(self._tracks.keys()):
                t = self._tracks[tid]
                t.lost += 1
                t.age += 1
                if t.lost > self.max_lost:
                    del self._tracks[tid]
            return {}

        det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        det_scores = [d[4] if len(d) > 4 else 0.5 for d in detections]

        n_dets = len(det_bboxes)
        n_tracks = len(self._tracks)

        # --- 贪心 IoU 匹配 ---
        matched_tids = set()
        matched_dets: Dict[int, int] = {}

        if n_tracks > 0 and n_dets > 0:
            iou_matrix = np.zeros((n_tracks, n_dets))
            tid_list = list(self._tracks.keys())
            for ti, tid in enumerate(tid_list):
                t = self._tracks[tid]
                for di in range(n_dets):
                    iou_matrix[ti, di] = _calc_iou(t.bbox, det_bboxes[di])

            # 贪心：每次取最大 IoU，匹配后移除
            remaining_tracks = set(range(n_tracks))
            remaining_dets = set(range(n_dets))
            while remaining_tracks and remaining_dets:
                best_iou = 0.0
                best_ti = None
                best_di = None
                for ti in remaining_tracks:
                    for di in remaining_dets:
                        if iou_matrix[ti, di] > best_iou:
                            best_iou = iou_matrix[ti, di]
                            best_ti = ti
                            best_di = di
                if best_iou < self.iou_threshold:
                    break
                tid = tid_list[best_ti]
                matched_tids.add(tid)
                matched_dets[best_di] = tid
                remaining_tracks.remove(best_ti)
                remaining_dets.remove(best_di)

        # --- 更新匹配成功的 track ---
        for di, tid in matched_dets.items():
            t = self._tracks[tid]
            t.bbox = det_bboxes[di]
            t.score = det_scores[di]
            t.hits += 1
            t.age += 1
            t.lost = 0
            if t.hits >= self.min_hits:
                t.confirmed = True

        # --- 未匹配的 track: lost++ ---
        for tid in list(self._tracks.keys()):
            if tid not in matched_tids:
                t = self._tracks[tid]
                t.lost += 1
                t.age += 1
                if t.lost > self.max_lost:
                    del self._tracks[tid]

        # --- 未匹配的 detection: 创建新 track ---
        for di in range(n_dets):
            if di not in matched_dets:
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = _IoUTrack(tid, det_bboxes[di], det_scores[di])

        # --- 返回活跃 bbox ---
        result = {}
        for tid, t in self._tracks.items():
            if t.confirmed or t.hits >= 1:
                result[tid] = t.bbox
        return result

    def get_memory(self) -> TrackMemory:
        """返回 TrackMemory 兼容层。"""
        return self._memory

    @property
    def active_count(self) -> int:
        return len(self._tracks)
