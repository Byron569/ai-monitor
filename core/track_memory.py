"""
TrackMemory — 长时轨迹记忆。匈牙利全局最优匹配。
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrackState:
    __slots__ = ("track_id", "bbox", "confidence", "created_at",
                 "last_seen", "age", "velocity", "status")

    def __init__(self, tid, bbox):
        self.track_id = tid
        self.bbox = bbox
        self.confidence = 0.9
        self.created_at = time.time()
        self.last_seen = time.time()
        self.age = 0
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self.status = "active"


class TrackMemory:
    def __init__(self, ttl_active=1.5, ttl_lost=1.5, max_tracks=30):
        self.ttl_active = ttl_active
        self.ttl_lost = ttl_lost
        self.max_tracks = max_tracks
        self._tracks: Dict[int, TrackState] = {}
        self._new_registrations: set = set()
        self._locked_matches: Dict[int, int] = {}  # tid → det_idx

    def update(self, active_tracks):
        now = time.time()
        active_ids = set(active_tracks.keys())
        for tid, bbox in active_tracks.items():
            if tid in self._tracks:
                ts = self._tracks[tid]
                ts.bbox = bbox
                ts.last_seen = now
                ts.age += 1
                ts.status = "active"
            else:
                self._tracks[tid] = TrackState(tid, bbox)
                logger.info(f"[MEMORY] track={tid} registered")
                self._new_registrations.add(tid)
        for tid in list(self._tracks.keys()):
            if tid not in active_ids:
                ts = self._tracks[tid]
                age = now - ts.last_seen
                if age > self.ttl_lost:
                    self._tracks.pop(tid, None)
                elif age > self.ttl_active and ts.status == "active":
                    ts.status = "lost"

    def match_hungarian(self, detections, max_dist=300):
        if not detections or not self._tracks:
            self._locked_matches.clear()
            return {}

        tracks = list(self._tracks.items())
        n_dets, n_tracks = len(detections), len(tracks)

        # --- 锁定确认：上一帧匹配仍有效 → 不解开 ---
        result = {}
        locked_tids = set()
        locked_dets = set()
        stale_locks = []
        for tid, last_det_idx in self._locked_matches.items():
            if last_det_idx < n_dets and tid in self._tracks:
                db = detections[last_det_idx]
                dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                ts = self._tracks[tid]
                tcx, tcy = (ts.bbox[0] + ts.bbox[2]) / 2, (ts.bbox[1] + ts.bbox[3]) / 2
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist <= max_dist:
                    result[last_det_idx] = tid
                    locked_tids.add(tid)
                    locked_dets.add(last_det_idx)
                else:
                    stale_locks.append(tid)
            else:
                stale_locks.append(tid)
        for tid in stale_locks:
            self._locked_matches.pop(tid, None)

        # --- 剩余 → 匈牙利 ---
        free_dets = [d for i, d in enumerate(detections) if i not in locked_dets]
        free_tracks = [(tid, ts) for tid, ts in tracks if tid not in locked_tids]

        if free_dets and free_tracks:
            nf = max(len(free_dets), len(free_tracks))
            cost = np.full((nf, nf), 1e6, dtype=np.float64)
            for i, db in enumerate(free_dets):
                dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                dw, dh = db[2] - db[0], db[3] - db[1]
                for j, (_tid, ts) in enumerate(free_tracks):
                    tcx, tcy = (ts.bbox[0] + ts.bbox[2]) / 2, (ts.bbox[1] + ts.bbox[3]) / 2
                    tw, th = ts.bbox[2] - ts.bbox[0], ts.bbox[3] - ts.bbox[1]
                    dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                    sdiff = abs(dw - tw) + abs(dh - th)
                    vx = ts.velocity[0] + ts.velocity[2]
                    if abs(vx) > 5:
                        expected_right = vx > 0
                        actual_right = dcx > tcx
                        if expected_right != actual_right:
                            dist += 200
                    c = dist + 2.0 * sdiff
                    if c <= max_dist:
                        cost[i, j] = c

            assignment = _hungarian(cost)
            for det_idx, track_idx in assignment.items():
                if det_idx < len(free_dets) and track_idx < len(free_tracks):
                    if cost[det_idx, track_idx] <= max_dist:
                        orig_det_idx = detections.index(free_dets[det_idx])
                        tid = free_tracks[track_idx][0]
                        result[orig_det_idx] = tid

        # --- 更新锁 ---
        self._locked_matches = {tid: di for di, tid in result.items()}
        return result

    def get_active(self):
        return {tid: ts.bbox for tid, ts in self._tracks.items() if ts.status == "active"}

    def get_lost(self):
        return {tid: ts for tid, ts in self._tracks.items() if ts.status == "lost"}

    @property
    def active_count(self):
        return sum(1 for ts in self._tracks.values() if ts.status == "active")

    @property
    def lost_count(self):
        return sum(1 for ts in self._tracks.values() if ts.status == "lost")

    def clear_lock(self, track_id: int) -> None:
        self._locked_matches.pop(track_id, None)

    def flush_new_registrations(self) -> set:
        regs = self._new_registrations.copy()
        self._new_registrations.clear()
        return regs


def _hungarian(cost: np.ndarray) -> dict:
    """手写匈牙利算法，返回 {row: col}。"""
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result = {}
    for j in range(1, n + 1):
        if p[j] > 0:
            result[p[j] - 1] = j - 1
    return result
