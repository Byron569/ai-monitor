"""
SCRFD Face Detector — 基于 InsightFace FaceAnalysis 的工业级人脸检测器。

接口契约:
    detector = SCRFDDetector(model_name="buffalo_s")
    detections = detector.detect(frame)
    # detections: List[(x1, y1, x2, y2, confidence, landmarks)]
    # landmarks: [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]

工程特性:
  - 分阶段耗时统计 (preprocess / inference / postprocess)
  - 滚动平均值 latency 报告
  - 可开关性能日志
  - backend 自动检测 (CUDA / CPU)
  - 输入标准化 (BGR float32, InsightFace 内部处理)
"""

import logging
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SCRFDDetector:
    """
    工业级 SCRFD 人脸检测器。

    参数:
        model_name:  InsightFace 模型包名 (默认 "buffalo_s")
        input_size:  检测输入尺寸 (默认 640)
        conf_threshold: 置信度阈值 (默认 0.5)
        device:      "cuda" 或 "cpu"
        verbose:     是否输出每帧耗时日志 (默认 False)
        stat_window: 滚动平均窗口大小 (默认 30 帧)

    使用示例:
        detector = SCRFDDetector(verbose=True)
        dets = detector.detect(frame)
        print(detector.stats_report())
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        input_size: int = 640,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cuda",
        verbose: bool = False,
        stat_window: int = 30,
    ):
        import insightface
        from insightface.app import FaceAnalysis

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.verbose = verbose
        self.device_str = device

        ctx_id = 0 if device == "cuda" else -1

        logger.info(f"[SCRFD] 加载模型: {model_name}")
        logger.info(f"[SCRFD] 输入尺寸: {input_size}x{input_size}")

        # 只加载检测模块
        self.app = FaceAnalysis(name=model_name, allowed_modules=["detection"])
        self.app.prepare(ctx_id=ctx_id, det_size=(input_size, input_size))

        # Backend 检测
        self._backend = "CPU"
        det_model = self.app.models.get("detection")
        if det_model is not None:
            session = getattr(det_model, "session", None)
            if session is not None:
                providers = session.get_providers()
                if "CUDAExecutionProvider" in providers:
                    self._backend = "CUDA"
        logger.info(f"[SCRFD] backend: {self._backend} | conf={conf_threshold}")

        # --- 性能统计 ---
        self._frame_count = 0
        self._stat_window = stat_window
        self._pre_times: deque = deque(maxlen=stat_window)
        self._inf_times: deque = deque(maxlen=stat_window)
        self._post_times: deque = deque(maxlen=stat_window)
        self._total_times: deque = deque(maxlen=stat_window)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, list]]:
        """
        检测人脸。

        Args:
            frame: BGR 图像, shape (H, W, 3), dtype=uint8

        Returns:
            List of (x1, y1, x2, y2, confidence, landmarks)
            landmarks: [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]
        """
        t0 = time.perf_counter()

        # --- Preprocess (InsightFace 内部处理: resize + normalize + BGR->RGB) ---
        # InsightFace 的 FaceAnalysis.get() 自动完成预处理
        # 无需手动处理，只需确保 frame 是 contiguous uint8 BGR
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        t1 = time.perf_counter()

        # --- Inference ---
        faces = self.app.get(frame)

        t2 = time.perf_counter()

        # --- Postprocess ---
        results = []
        for face in faces:
            if face.det_score < self.conf_threshold:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            conf = float(face.det_score)
            kps = face.kps
            landmarks = [
                (int(kps[i][0]), int(kps[i][1])) for i in range(5)
            ]
            results.append((x1, y1, x2, y2, conf, landmarks))

        results.sort(key=lambda r: r[4], reverse=True)

        t3 = time.perf_counter()

        # --- 记录耗时 ---
        pre_ms = (t1 - t0) * 1000
        inf_ms = (t2 - t1) * 1000
        post_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        self._pre_times.append(pre_ms)
        self._inf_times.append(inf_ms)
        self._post_times.append(post_ms)
        self._total_times.append(total_ms)

        if self.verbose and self._frame_count % self._stat_window == 0:
            logger.info(
                f"[SCRFD] pre:{self._avg_ms(self._pre_times):.1f}ms | "
                f"infer:{self._avg_ms(self._inf_times):.1f}ms | "
                f"post:{self._avg_ms(self._post_times):.1f}ms | "
                f"total:{self._avg_ms(self._total_times):.1f}ms"
            )

        self._frame_count += 1
        return results

    # ------------------------------------------------------------------
    # 性能统计
    # ------------------------------------------------------------------

    def stats_report(self) -> str:
        """返回性能报告字符串。"""
        if self._frame_count == 0:
            return "[SCRFD] no data"
        return (
            f"[SCRFD] frames:{self._frame_count} | "
            f"pre:{self._avg_ms(self._pre_times):.1f}ms | "
            f"infer:{self._avg_ms(self._inf_times):.1f}ms | "
            f"post:{self._avg_ms(self._post_times):.1f}ms | "
            f"total:{self._avg_ms(self._total_times):.1f}ms | "
            f"backend:{self._backend}"
        )

    @property
    def avg_latency_ms(self) -> float:
        """返回平均总延迟 (ms)。"""
        return self._avg_ms(self._total_times)

    @property
    def backend(self) -> str:
        """返回当前 backend。"""
        return self._backend

    @staticmethod
    def _avg_ms(dq: deque) -> float:
        if not dq:
            return 0.0
        return sum(dq) / len(dq)
