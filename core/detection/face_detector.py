"""
人脸检测模块 — 封装 YOLOv8-face / OpenCV Haar 级联，提供统一的人脸检测接口。

优先使用 YOLOv8-face 模型（需要联网下载权重）。
若 YOLO 加载失败，自动回退到 OpenCV Haar 级联（无需联网，已内置）。
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np


# 检测结果类型：(x1, y1, x2, y2, confidence)
Detection = Tuple[int, int, int, int, float]


class HaarCascadeFaceDetector:
    """
    OpenCV Haar 级联人脸检测器。

    使用 OpenCV 自带的 haarcascade_frontalface_default.xml，无需联网下载。
    作为 YOLOv8-face 的离线备选方案。
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"无法加载 Haar 级联文件: {cascade_path}")
        print(f"[FaceDetector] 使用 Haar 级联检测器 (无需联网)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )
        faces: List[Detection] = []
        for (x, y, w, h) in rects:
            faces.append((x, y, x + w, y + h, 0.9))
        faces.sort(key=lambda d: d[4], reverse=True)
        return faces


class FaceDetector:
    """
    人脸检测器封装类 — 自动选择后端。

    优先使用 YOLOv8-face 模型（需联网下载权重）。
    若 YOLO 加载失败，自动回退到 OpenCV Haar 级联（离线可用）。

    使用示例：
      detector = FaceDetector()
      faces = detector.detect(frame)
      for (x1, y1, x2, y2, conf) in faces:
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """

    AVAILABLE_MODELS = {
        "nano": "yolov8n-face.pt",
        "small": "yolov8s-face.pt",
        "medium": "yolov8m-face.pt",
        "yolov8-face": "yolov8n-face.pt",
    }

    def __init__(
        self,
        model_variant: str = "nano",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        image_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self._backend = None

        # 尝试加载 YOLOv8-face
        try:
            from ultralytics import YOLO
            import os
            if model_path is not None:
                self.model_path = model_path
            else:
                if model_variant not in self.AVAILABLE_MODELS:
                    raise ValueError(
                        f"不支持的模型变体: {model_variant}。"
                        f"可选: {list(self.AVAILABLE_MODELS.keys())}"
                    )
                # 优先查找 models/ 目录，其次是当前目录
                filename = self.AVAILABLE_MODELS[model_variant]
                project_models = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models", filename
                )
                self.model_path = project_models if os.path.exists(project_models) else filename

            print(f"[FaceDetector] 正在加载 YOLO 模型: {self.model_path} ...")
            self._yolo = YOLO(self.model_path)

            import torch
            if device is not None:
                self._yolo.to(device)
                self._use_fp16 = (device == "cuda")
            elif torch.cuda.is_available():
                self._yolo.to("cuda")
                self._use_fp16 = True
                print("[FaceDetector] 已启用 GPU + FP16 推理")
            else:
                self._use_fp16 = False
                print("[FaceDetector] 未检测到 GPU，使用 CPU")

            self._backend = "yolo"
            print(f"[FaceDetector] YOLO 模型加载完成 (variant={model_variant}, "
                  f"conf={confidence_threshold}, imgsz={image_size})")
        except Exception as e:
            print(f"[FaceDetector] YOLO 加载失败 ({e})，回退到 Haar 级联")
            self._haar = HaarCascadeFaceDetector(confidence_threshold)
            self._backend = "haar"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._backend == "yolo":
            results = self._yolo(frame, imgsz=self.image_size, verbose=False, half=self._use_fp16)
            result = results[0]
            faces: List[Detection] = []
            if result.boxes is None:
                return faces
            for box in result.boxes:
                data = box.data.cpu().numpy()[0] if box.data.is_cuda else box.data.numpy()[0]
                x1, y1, x2, y2 = map(int, data[:4])
                confidence = float(data[4])
                if confidence < self.confidence_threshold:
                    continue
                faces.append((x1, y1, x2, y2, confidence))
            faces.sort(key=lambda d: d[4], reverse=True)
            return faces
        else:
            return self._haar.detect(frame)
