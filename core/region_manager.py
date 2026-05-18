"""
RegionManager — 区域系统。

支持: 禁区(restricted)、关注区(interest)、入口区(entry)。
判断 bbox 中心是否在多边形内（射线法）。

用法:
    rm = RegionManager()
    rm.add_zone("restricted", "server_room", [(100,100),(300,100),(300,300),(100,300)])
    inside, zone_name = rm.check_inside(bbox)
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RegionManager:
    def __init__(self):
        self._zones: Dict[str, List[Tuple[str, List[Tuple[int, int]]]]] = {}
        self._track_inside: Dict[int, Dict[str, bool]] = {}

    def add_zone(self, zone_type, name, points):
        if zone_type not in self._zones:
            self._zones[zone_type] = []
        self._zones[zone_type].append((name, points))
        logger.info(f"[REGION] zone added: {name} ({zone_type})")

    def check_inside(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for zone_type, zones in self._zones.items():
            for name, points in zones:
                if _point_in_polygon(cx, cy, points):
                    return True, zone_type, name
        return False, "", ""

    def check_entry(self, track_id: int, bbox) -> tuple:
        """检测新进入（仅状态变化时返回 True）。"""
        inside, ztype, zname = self.check_inside(bbox)
        if track_id not in self._track_inside:
            self._track_inside[track_id] = {}
        prev = self._track_inside[track_id].get(ztype, False)
        self._track_inside[track_id][ztype] = inside
        if inside and not prev:
            return True, ztype, zname
        return False, "", ""

    def has_zone_type(self, zone_type: str) -> bool:
        return zone_type in self._zones and len(self._zones[zone_type]) > 0


def _point_in_polygon(px: float, py: float, polygon: List[Tuple[int, int]]) -> bool:
    """射线法判断点是否在多边形内。"""
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside
