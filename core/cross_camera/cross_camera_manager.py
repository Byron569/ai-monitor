"""CrossCameraManager — 跨摄像头交叉验证协调器。支持 single / dual / multi 模式。"""

import logging
from typing import Dict, Optional

from core.cross_camera.matcher import match_cross_camera, merge_tracking_ids

logger = logging.getLogger(__name__)


class CrossCameraManager:

    def __init__(self, mode: str = "single", hist_threshold: float = 0.5):
        self.mode = mode
        self.hist_threshold = hist_threshold
        self._global_id_map: dict = {}
        self._pid_to_global: dict = {}
        self._fall_state_store: Dict[str, dict] = {}

    def is_cross_check_enabled(self) -> bool:
        return self.mode in ("dual", "multi")

    def match_and_merge(self, camera_results: Dict[str, list]) -> Dict[str, dict]:
        if self.mode == "single":
            return {}
        result = {}
        cam_ids = list(camera_results.keys())
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                cid_a, cid_b = cam_ids[i], cam_ids[j]
                tracked_a = camera_results.get(cid_a, [])
                tracked_b = camera_results.get(cid_b, [])
                if not tracked_a or not tracked_b:
                    continue
                pairs = match_cross_camera(tracked_a, tracked_b, self.hist_threshold)
                pid_to_global = merge_tracking_ids(
                    tracked_a, tracked_b, pairs, self._global_id_map, self._pid_to_global
                )
                self._pid_to_global = pid_to_global
        for cid in cam_ids:
            result[cid] = {}
            for person in camera_results.get(cid, []):
                pid = person["pid"]
                gid = self._pid_to_global.get(pid)
                if gid is not None:
                    result[cid][pid] = gid
        return result

    def get_dual_cam_fall(self, camera_id: str, pid: int, all_results: Dict[str, list]) -> Optional[bool]:
        if self.mode == "single":
            return None
        gid = self._pid_to_global.get(pid)
        if gid is None:
            return None
        for cid, persons in all_results.items():
            if cid == camera_id:
                continue
            for p in persons:
                other_gid = self._pid_to_global.get(p["pid"])
                if other_gid == gid:
                    return p.get("fall_detected", False)
        return None

    def cleanup(self, active_pids: set):
        stale = [k for k in self._global_id_map
                 if k[0] not in active_pids and k[1] not in active_pids]
        for k in stale:
            del self._global_id_map[k]
