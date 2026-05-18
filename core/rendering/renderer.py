"""
渲染模块 — 统一管理画面叠加逻辑。

将绘制逻辑从 Pipeline 中剥离，遵循单一职责原则。
所有绘制方法原地修改 frame，不返回复制。
"""

import cv2
import numpy as np


# 默认配色方案（B, G, R）
COLOR_IDENTIFIED = (0, 255, 0)      # 绿色 — 已识别
COLOR_UNKNOWN = (128, 128, 128)      # 灰色 — 未识别
COLOR_FPS = (0, 255, 255)            # 黄色 — FPS
COLOR_CONFIDENCE = (0, 255, 0)       # 绿色 — 置信度
COLOR_FALL = (0, 0, 255)             # 红色 — 摔倒
COLOR_POTENTIAL = (0, 165, 255)      # 橙色 — 疑似摔倒
COLOR_SKELETON = (255, 204, 102)     # 浅蓝 — 骨架
COLOR_KEYPOINT = (102, 204, 255)     # 金色 — 关键点
COLOR_BODY = (180, 50, 0)            # 深蓝 — 人体框

FONT = cv2.FONT_HERSHEY_SIMPLEX

# YOLOv8-Pose 17 关键点骨架连接 (COCO format)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),                 # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),     # 腿部
]


class Renderer:
    """
    画面渲染器。

    职责：
      1. 绘制人脸检测框 + 身份标签
      2. 绘制摔倒检测骨架 + 状态标签
      3. 绘制 FPS 叠加
      4. 统一配色方案与字体

    设计原则：
      - 所有方法原地修改 frame
      - 配色集中管理，方便后续做主题切换
      - 与检测/识别逻辑完全解耦
    """

    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        fps_font_scale: float = 0.8,
    ) -> None:
        self.font_scale = font_scale
        self.thickness = thickness
        self.fps_font_scale = fps_font_scale

    # ------------------------------------------------------------------
    # 人脸识别绘制
    # ------------------------------------------------------------------

    def draw_face_identity(
        self,
        frame: np.ndarray,
        bbox: tuple,
        name: str = "Unknown",
        similarity: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        x1, y1, x2, y2 = bbox

        is_identified = (name != "Unknown")
        color = COLOR_IDENTIFIED if is_identified else COLOR_UNKNOWN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        if is_identified:
            label = f"{name} ({similarity:.2f})"
        else:
            label = name

        (tw, th), baseline = cv2.getTextSize(label, FONT, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

        cv2.putText(
            frame, label, (x1 + 3, y1 - 6),
            FONT, self.font_scale, (255, 255, 255), self.thickness,
        )

    # ------------------------------------------------------------------
    # 摔倒检测绘制
    # ------------------------------------------------------------------

    def draw_skeleton(self, frame: np.ndarray, keypoints: list) -> None:
        """绘制 17 关键点骨架。keypoints: list of [x, y] (pixel coords)。"""
        pts = [(int(kp[0]), int(kp[1])) for kp in keypoints]

        # 骨骼连线
        for a, b in SKELETON_EDGES:
            if a >= len(pts) or b >= len(pts):
                continue
            pt_a, pt_b = pts[a], pts[b]
            if pt_a[0] < 0 or pt_a[1] < 0 or pt_b[0] < 0 or pt_b[1] < 0:
                continue
            cv2.line(frame, pt_a, pt_b, COLOR_SKELETON, 2)

        # 关键点
        for pt in pts:
            if pt[0] <= 0 or pt[1] <= 0:
                continue
            if pt[0] < 0 or pt[1] < 0:
                continue
            cv2.circle(frame, pt, 3, COLOR_KEYPOINT, -1)

    def draw_body_bbox(self, frame: np.ndarray, bbox: tuple) -> None:
        """绘制人体检测框 (区别于绿色人脸框)。"""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BODY, self.thickness)

    def draw_fall_status(
        self,
        frame: np.ndarray,
        bbox: tuple,
        fall_state: str = "",
        confidence: float = 0.0,
    ) -> None:
        """在 bbox 上方绘制摔倒状态标签。"""
        if not fall_state:
            return

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if fall_state == "FALL":
            label = f"FALL ({confidence:.2f})"
            color = COLOR_FALL
        elif fall_state == "Potential Fall":
            label = "WARNING: Potential Fall"
            color = COLOR_POTENTIAL
        else:
            label = fall_state
            color = COLOR_IDENTIFIED  # 绿色 Normal

        (tw, th), _ = cv2.getTextSize(label, FONT, self.font_scale + 0.1, self.thickness + 1)
        label_y = max(y1 - 40, 20)
        cv2.rectangle(frame, (x1, int(label_y - th - 8)), (x1 + tw + 6, int(label_y + 2)), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, int(label_y - 4)),
            FONT, self.font_scale + 0.1, (255, 255, 255), self.thickness + 1,
        )

    # ------------------------------------------------------------------
    # FPS / 系统信息
    # ------------------------------------------------------------------

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_str, (10, 30),
            FONT, self.fps_font_scale, COLOR_FPS, self.thickness + 1,
        )

    def draw_system_info(
        self,
        frame: np.ndarray,
        lines: list,
        start_y: int = 60,
        line_spacing: int = 30,
    ) -> None:
        for i, line in enumerate(lines):
            y = start_y + i * line_spacing
            cv2.putText(
                frame, line, (10, y),
                FONT, self.font_scale, (255, 255, 255), self.thickness,
            )
