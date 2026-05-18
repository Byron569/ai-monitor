"""
后端工厂函数
根据 backend 名称创建对应的推理后端实例
"""

import logging

logger = logging.getLogger(__name__)


def create_backend(
    backend="ultralytics",
    model_path=None,
    device="cpu",
    conf=0.35,
    tracker="bytetrack.yaml",
    input_size=640,
    enable_tracking=True,
):
    """
    创建推理后端实例

    Args:
        backend: 后端名称 ("ultralytics", "onnx", 预留 "openvino", "ncnn")
        model_path: 模型文件路径
        device: 推理设备
        conf: 检测置信度阈值
        tracker: 跟踪器配置（仅 ultralytics 后端）
        input_size: 模型输入分辨率
        enable_tracking: 是否启用 ByteTracker（仅 ultralytics 后端）

    Returns:
        BaseInferenceBackend 实例

    Raises:
        ValueError: 不支持的后端名称
    """
    backend_name = backend.lower()

    if backend_name == "ultralytics":
        from plugins.fall_engine.backends.ultralytics_backend import UltralyticsBackend

        return UltralyticsBackend(
            model_path=model_path,
            device=device,
            conf=conf,
            tracker=tracker,
            input_size=input_size,
            enable_tracking=enable_tracking,
        )

    elif backend_name == "onnx":
        from plugins.fall_engine.backends.onnx_backend import ONNXBackend

        return ONNXBackend(
            model_path=model_path,
            device=device,
            conf=conf,
            input_size=input_size,
        )

    elif backend_name == "openvino":
        raise NotImplementedError("OpenVINO backend 尚未实现")

    elif backend_name == "ncnn":
        from plugins.fall_engine.backends.ncnn_backend import NCNNBackend

        return NCNNBackend(
            model_path=model_path,
            device=device,
            conf=conf,
            input_size=input_size,
        )

    else:
        raise ValueError(
            f"不支持的后端: '{backend}'。"
            f"当前支持: ultralytics, onnx, ncnn"
        )
