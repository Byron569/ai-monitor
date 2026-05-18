"""
FPS 工具模块 — 实时帧率统计。

使用滑动窗口平均法，避免单帧抖动导致 FPS 显示剧烈跳动。
"""

import time
from collections import deque


class FPS:
    """
    实时 FPS 统计类。

    职责：
      1. 每帧调用 update() 记录时间戳
      2. 基于滑动窗口返回平滑 FPS

    使用示例：
      fps = FPS(window_size=30)
      while True:
          ...  # 一帧的完整处理
          fps.update()
          current_fps = fps.get_fps()
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        初始化 FPS 统计器。

        Args:
            window_size: 滑动窗口大小（帧数）。值越大 FPS 越平滑但响应越慢。
                         默认 30 帧，适合实时显示。
        """
        self.window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._last_timestamp: float = time.time()

    def update(self) -> None:
        """
        记录当前帧的时间。

        每处理完一帧后调用一次。
        """
        now = time.time()
        self._timestamps.append(now)
        self._last_timestamp = now

    def get_fps(self) -> float:
        """
        返回当前平滑 FPS 值。

        Returns:
            float: 每秒帧数。累积的帧数不够时返回 0.0。

        原理：
          FPS = 窗口内的帧数 / 窗口时间跨度
          例如：30 帧跨越 1 秒 → FPS = 30.0
        """
        if len(self._timestamps) < 2:
            return 0.0

        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / duration

    def get_fps_str(self) -> str:
        """
        返回格式化 FPS 字符串，方便直接叠加到画面。

        示例: "FPS: 29.8"
        """
        fps = self.get_fps()
        return f"FPS: {fps:.1f}"
