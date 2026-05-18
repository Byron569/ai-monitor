"""
推理后端子包
提供多种推理后端实现和工厂函数
"""

from plugins.fall_engine.backends.base import BaseInferenceBackend
from plugins.fall_engine.backends.factory import create_backend
from plugins.fall_engine.backends.postprocess import (
    ultralytics_results_to_detections,
    onnx_outputs_to_detections,
)
from plugins.fall_engine.backends.ncnn_backend import NCNNBackend

__all__ = [
    "BaseInferenceBackend",
    "create_backend",
    "ultralytics_results_to_detections",
    "onnx_outputs_to_detections",
    "NCNNBackend",
]
