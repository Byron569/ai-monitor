"""
推理后端抽象基类
所有后端必须实现此接口
"""

from abc import ABC, abstractmethod


class BaseInferenceBackend(ABC):
    """推理后端抽象基类

    所有后端（ultralytics / onnx / openvino / ncnn）必须实现 infer / warmup / close。
    统一返回 detection schema，不返回任何框架特有对象。
    """

    @abstractmethod
    def infer(self, frame):
        """
        对单帧图像执行推理

        Args:
            frame: OpenCV BGR 图像 (numpy array, HWC, uint8)

        Returns:
            list[dict]: 统一检测结果列表，每个元素格式:
                {
                    "bbox": [x1, y1, x2, y2],      # list, 像素坐标
                    "score": float,                   # 检测置信度
                    "class_id": 0,                    # 固定为 person
                    "track_id": int | None,           # 跟踪 ID，None 表示未跟踪
                    "keypoints": [[x, y, conf], ...]  # 17 个关键点
                }
        """
        pass

    @abstractmethod
    def warmup(self):
        """预热：执行一次空推理以初始化引擎"""
        pass

    @abstractmethod
    def close(self):
        """释放后端资源"""
        pass
