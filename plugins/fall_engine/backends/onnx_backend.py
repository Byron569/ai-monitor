"""
ONNX Runtime 推理后端
使用 onnxruntime 进行推理，不依赖 PyTorch / ultralytics
"""

import logging
import numpy as np
import cv2

from plugins.fall_engine.backends.base import BaseInferenceBackend
from plugins.fall_engine.backends.postprocess import onnx_outputs_to_detections

logger = logging.getLogger(__name__)


class ONNXBackend(BaseInferenceBackend):
    """基于 ONNX Runtime 的推理后端

    支持 CPUExecutionProvider 和 CUDAExecutionProvider。
    不依赖 PyTorch / ultralytics，适合边缘设备部署。
    """

    def __init__(
        self,
        model_path,
        device="cpu",
        conf=0.35,
        input_size=640,
    ):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.input_size = input_size

        self._session = None
        self._input_name = None
        self._output_names = None
        self._input_shape = None
        self._providers = None
        self._init_session()

    def _init_session(self):
        import onnxruntime as ort

        providers = []
        if self.device.startswith("cuda"):
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
            self._providers = self._session.get_providers()
        except Exception as e:
            logger.error(f"加载 ONNX 模型失败: {e}")
            raise

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._input_shape = self._session.get_inputs()[0].shape
        logger.info(f"ONNXBackend: providers={self._providers}, input={self._input_shape}")

    def infer(self, frame):
        """执行 ONNX 推理，返回统一检测列表"""
        if self._session is None:
            return []

        # 预处理
        input_tensor = self._preprocess(frame)

        # 推理
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # 后处理
        return onnx_outputs_to_detections(outputs, frame.shape[:2], conf_threshold=self.conf, input_size=self.input_size)

    def _preprocess(self, frame):
        """预处理：BGR→RGB, resize, normalize, CHW, float32"""
        # Resize
        h, w = frame.shape[:2]
        scale = self.input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (new_w, new_h))

        # 填充到 input_size 的正方形
        # 但 YOLOv8-pose 通常使用 letterbox 或直接 resize
        # 这里使用直接 resize 到 input_size（需与导出时一致）
        img = cv2.resize(img, (self.input_size, self.input_size))

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # CHW
        img = img.transpose(2, 0, 1)

        # Batch dimension
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def warmup(self):
        """预热推理"""
        dummy = np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8)
        _ = self.infer(dummy)
        logger.debug("ONNXBackend warmup done")

    def close(self):
        """释放资源"""
        self._session = None
