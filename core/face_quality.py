"""
FaceQualityFilter — 人脸质量过滤器。

在 ArcFace 识别前快速评估人脸 crop 质量，
避免低质量人脸浪费 GPU 推理资源。

规则：
  1. bbox 宽/高 < min_face_size → 拒绝
  2. bbox 越界 → 拒绝
  3. 长宽比异常 (w/h > 2.5 or h/w > 2.5) → 拒绝
  4. Laplacian 模糊度 < blur_threshold → 拒绝
  5. 综合分 < min_quality_score → 拒绝
"""

import cv2
import numpy as np


class FaceQualityResult:
    __slots__ = ("passed", "score", "reason", "blur_score", "face_size")

    def __init__(self, passed, score, reason, blur_score, face_size):
        self.passed = passed
        self.score = score
        self.reason = reason
        self.blur_score = blur_score
        self.face_size = face_size

    def __repr__(self):
        return f"FaceQuality(pass={self.passed}, score={self.score:.2f}, reason={self.reason})"


class FaceQualityFilter:
    """
    轻量人脸质量评估器。

    参数:
        min_face_size:   最小人脸边长 (px), 默认 48
        blur_threshold:  Laplacian 方差最低值, 默认 80
        min_quality_score: 综合分数阈值 [0,1], 默认 0.55
    """

    def __init__(
        self,
        min_face_size: int = 48,
        blur_threshold: float = 80.0,
        min_quality_score: float = 0.55,
    ):
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.min_quality_score = min_quality_score

    def evaluate(
        self,
        frame: np.ndarray,
        bbox: tuple,
        landmarks=None,
    ) -> FaceQualityResult:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        fh, fw = frame.shape[:2]

        # Rule 1: minimum size
        if w < self.min_face_size or h < self.min_face_size:
            return FaceQualityResult(False, 0.0, f"too_small({w}x{h})", 0.0, (w, h))

        # Rule 2: out of bounds
        if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh or w <= 0 or h <= 0:
            return FaceQualityResult(False, 0.0, "out_of_bounds", 0.0, (w, h))

        # Rule 3: aspect ratio
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > 2.5:
            return FaceQualityResult(False, 0.0, f"bad_aspect({ratio:.1f})", 0.0, (w, h))

        # Rule 4: blur detection (Laplacian variance)
        face_crop = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
        if face_crop.size == 0:
            return FaceQualityResult(False, 0.0, "empty_crop", 0.0, (w, h))
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < self.blur_threshold:
            return FaceQualityResult(False, 0.0, f"blurry({blur_score:.0f})", blur_score, (w, h))

        # Rule 5: composite score
        size_score = min(w * h / (80 * 80), 1.0)
        blur_norm = min(blur_score / 500.0, 1.0)
        score = 0.5 * size_score + 0.5 * blur_norm

        if score < self.min_quality_score:
            return FaceQualityResult(False, score, f"low_score({score:.2f})", blur_score, (w, h))

        return FaceQualityResult(True, score, "ok", blur_score, (w, h))
