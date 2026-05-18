"""
Motion Gate — 基于帧差法的移动检测器。

极轻量：只需两张灰度图的 abs diff → mean score。
无 AI 模型，纯 numpy 运算。

用法:
    gate = MotionGate(threshold=5.0)
    active, score = gate.check(frame)
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionGate:
    """
    基于帧差法的运动检测。

    参数:
        threshold: 帧差均值阈值（越大越不敏感，默认 5.0）
        history:   保留多少帧做背景估计（>1 启用 MOG 模式，可选）

    用法:
        gate = MotionGate(threshold=5.0)
        for frame in camera:
            active, score = gate.check(frame)
            if active:
                run_detection()
    """

    def __init__(
        self,
        threshold: float = 5.0,
        history: int = 0,
    ):
        self.threshold = threshold
        self._prev_gray: np.ndarray | None = None
        self._score: float = 0.0
        self._active: bool = False
        self._bg_sub: cv2.BackgroundSubtractorMOG2 | None = None

        if history > 1:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=16, detectShadows=False
            )
            logger.info(f"[MOTION] 使用 MOG2 背景建模 (history={history})")
        else:
            logger.info(f"[MOTION] 使用帧差法 (threshold={threshold})")

    def check(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        检查当前帧是否有运动。

        Returns:
            (active, score): active 为 True 表示有运动
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._bg_sub is not None:
            fg_mask = self._bg_sub.apply(gray)
            self._score = float(fg_mask.mean())
        else:
            if self._prev_gray is None:
                self._prev_gray = gray.copy()
                self._score = 0.0
                self._active = False
                return False, 0.0

            diff = cv2.absdiff(gray, self._prev_gray)
            self._score = float(diff.mean())
            self._prev_gray = gray

        self._active = self._score >= self.threshold
        return self._active, self._score

    @property
    def score(self) -> float:
        return self._score

    @property
    def active(self) -> bool:
        return self._active
