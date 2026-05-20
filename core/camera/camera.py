"""
摄像头模块 — 封装 OpenCV VideoCapture，提供统一的摄像头读取接口。

适用场景：
  - USB 摄像头（Windows / Linux）
  - 手机虚拟摄像头（DroidCam / Iriun 等）
  - 后续可扩展 GStreamer 管道（Jetson 适配）
"""

import cv2
from typing import Optional, Tuple


class Camera:
    """
    摄像头封装类。

    职责：
      1. 打开指定索引的摄像头
      2. 按指定分辨率读取帧
      3. 释放摄像头资源

    使用示例：
      cam = Camera(index=0, width=1280, height=720)
      ret, frame = cam.read()
      cam.release()
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        api: Optional[int] = None,
    ) -> None:
        """
        初始化摄像头。

        Args:
            index:   摄像头设备索引。0=内置/第一个USB摄像头。
            width:   期望的采集宽度（像素），默认 640。
            height:  期望的采集高度（像素），默认 480。
            api:     OpenCV 后端 API 偏好（cv2.CAP_DSHOW / cv2.CAP_V4L2 等），
                    为 None 时自动选择。

        Raises:
            RuntimeError: 摄像头无法打开时抛出。
        """
        self.index = index
        self.width = width
        self.height = height

        # 在 Windows 上强制使用 DirectShow（MSMF 稳定性差）
        if api is not None:
            self.cap = cv2.VideoCapture(index, api)
        else:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 (index={index})。"
                f"请确认摄像头已连接且未被其他程序占用。"
            )

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 读取实际分辨率（某些摄像头可能不支持设置的尺寸）
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] 摄像头已打开 (index={index}, {actual_w:.0f}x{actual_h:.0f})")

        # 虚拟摄像头（DroidCam/Iriun等）需要预热几帧
        for _ in range(30):
            self.cap.read()

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        读取一帧画面。

        Returns:
            (ret, frame):
            - ret:   是否成功读取（bool）
            - frame: BGR 格式的帧（numpy.ndarray），失败时为 None
        """
        return self.cap.read()

    def release(self) -> None:
        """释放摄像头资源。"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[Camera] 摄像头已释放")

    def __enter__(self) -> "Camera":
        """上下文管理器入口，支持 with Camera(...) as cam: 语法。"""
        return self

    def __exit__(self, *args) -> None:
        """上下文管理器出口，自动释放资源。"""
        self.release()
