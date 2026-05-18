"""
Ultralytics 推理后端
封装 YOLO.track() 调用，返回统一 detection schema
"""

import logging
import numpy as np
import cv2

from plugins.fall_engine.backends.base import BaseInferenceBackend
from plugins.fall_engine.backends.postprocess import ultralytics_results_to_detections

logger = logging.getLogger(__name__)


class UltralyticsBackend(BaseInferenceBackend):
    """基于 Ultralytics YOLO 的推理后端"""

    def __init__(
        self,
        model_path,
        device="cpu",
        conf=0.35,
        tracker="bytetrack.yaml",
        input_size=640,
        enable_tracking=True,
    ):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.tracker = tracker
        self.input_size = input_size
        self.enable_tracking = enable_tracking

        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics 未安装，请: pip install ultralytics")

        try:
            self._model = YOLO(self.model_path)
        except Exception:
            logger.warning(f"无法加载模型 {self.model_path}，尝试自动下载")
            YOLO("yolov8n-pose.pt")
            self._model = YOLO(self.model_path)
        except Exception:
            logger.warning(f"无法加载模型 {self.model_path}，尝试自动下载")
            YOLO("yolov8n-pose.pt")
            self._model = YOLO(self.model_path)

    def infer(self, frame):
        """执行推理，返回统一检测列表"""
        if self._model is None:
            return []

        h, w = frame.shape[:2]

        # 如果输入尺寸和帧尺寸不同，先 resize
        if self.input_size != max(w, h):
            scale = self.input_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        if self.enable_tracking:
            results = self._model.track(
                frame,
                conf=self.conf,
                persist=True,
                tracker=self.tracker,
                verbose=False,
            )[0]
        else:
            results = self._model(frame, conf=self.conf, verbose=False)[0]

        return ultralytics_results_to_detections(results)

    def warmup(self):
        """预热推理"""
        dummy = np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8)
        _ = self.infer(dummy)
        logger.debug("UltralyticsBackend warmup done")

    def close(self):
        """释放资源"""
        self._model = None
