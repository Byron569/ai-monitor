"""
Frame Scheduler — 自适应检测调度器。

决策逻辑:
  motion_score > 10 (快速运动) → 每1帧检测
  motion_score > 3  (正常运动) → 每2帧检测  
  motion_score <= 3 (缓慢/静止) → 每5帧检测 (或 force_interval)
"""

import logging

logger = logging.getLogger(__name__)


class FrameScheduler:
    def __init__(self, detection_interval=2, force_interval=15):
        self.detection_interval = detection_interval
        self.force_interval = force_interval

    def should_detect(self, frame_count, motion_active, motion_score=0):
        if not motion_active:
            if frame_count % self.force_interval == 0:
                return True, "force"
            return False, "no motion"

        # 自适应间隔
        if motion_score > 10:
            interval = 1
            reason = "fast"
        elif motion_score > 3:
            interval = 2
            reason = "normal"
        else:
            interval = 5
            reason = "slow"

        if frame_count % interval == 0:
            return True, f"detect({reason})"
        return False, f"skip({reason})"
