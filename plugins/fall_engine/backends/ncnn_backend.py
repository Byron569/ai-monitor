"""
NCNN 推理后端
ARM NEON 原生加速, 用于树莓派 / ARM 边缘设备
"""

import logging
import numpy as np
import cv2

from plugins.fall_engine.backends.base import BaseInferenceBackend
from plugins.fall_engine.backends.postprocess import onnx_outputs_to_detections

logger = logging.getLogger(__name__)


class NCNNBackend(BaseInferenceBackend):
    """基于 Tencent NCNN 的推理后端

    ARM NEON 手写汇编优化, Winograd 3x3 卷积加速 (~1.8x)。
    树莓派 5 上 YOLOv8n-pose 320px 约 35ms (vs ONNX 100ms)。
    """

    def __init__(self, model_path, device="cpu", conf=0.35, input_size=640):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.input_size = input_size

        self._net = None
        self._input_name = "in0"
        self._init_net()

    def _init_net(self):
        try:
            import ncnn
        except ImportError:
            raise ImportError(
                "ncnn 未安装。安装: pip install ncnn\n"
                "树莓派: pip install ncnn (ARM aarch64/armv7l wheel 可用)"
            )

        self._net = ncnn.Net()
        self._net.opt.use_vulkan_compute = False
        self._net.opt.use_fp16_arithmetic = True
        self._net.opt.use_packing_layout = True
        self._net.opt.num_threads = 4

        param_path = self.model_path + ".param"
        bin_path = self.model_path + ".bin"

        ret = self._net.load_param(param_path)
        if ret != 0:
            raise RuntimeError(f"加载 NCNN param 失败: {param_path}")
        ret = self._net.load_model(bin_path)
        if ret != 0:
            raise RuntimeError(f"加载 NCNN bin 失败: {bin_path}")

        logger.info(f"NCNNBackend: model={self.model_path} threads={self._net.opt.num_threads}")

    def infer(self, frame):
        if self._net is None:
            return []

        input_tensor = self._preprocess(frame)

        import ncnn
        ex = self._net.create_extractor()
        ex.input(self._input_name, ncnn.Mat(input_tensor))

        ret, out_mat = ex.extract("output0")
        if ret != 0:
            return []

        out_np = np.array(out_mat).reshape(1, 56, -1)

        return onnx_outputs_to_detections(out_np, frame.shape[:2], conf_threshold=self.conf)

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def warmup(self):
        dummy = np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8)
        _ = self.infer(dummy)
        logger.debug("NCNNBackend warmup done")

    def close(self):
        self._net = None
