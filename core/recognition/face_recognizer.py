"""
人脸识别模块 — 封装 InsightFace ArcFace，提供 embedding 提取与身份比对接口。

架构决策：
  InsightFace 的 FaceAnalysis 内置了 RetinaFace 检测器。
  但我们使用 YOLOv8 做人脸检测，因此 FaceAnalysis 的检测能力仅用于：
    1. 在 YOLO 裁剪的小尺寸人脸 ROI 上做二次精确定位（开销极小）
    2. 检测 5 个关键点 → 仿射对齐 → ArcFace 提取 512 维 embedding

  这保证了人脸对齐的精度（ArcFace 对对齐质量敏感），同时 YOLOv8 负责
  画面级的大范围搜索，两者分工明确。

接口：
  recognizer = FaceRecognizer(device="cuda")
  embedding = recognizer.get_embedding(face_crop)      # np.ndarray (512,)
  name, score = recognizer.recognize(face_crop, db)     # ("ZhangSan", 0.85)
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import logging
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# InsightFace 是可选依赖 — 首次导入时会自动下载 buffalo_l 模型包
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class FaceRecognizer:
    """
    InsightFace ArcFace 识别器封装。

    职责：
      1. 加载 InsightFace buffalo_l 模型包（ArcFace w600k_r50）
      2. 对 YOLO 裁剪的人脸 ROI 提取 512 维归一化 embedding
      3. 计算余弦相似度
      4. 与 FaceDatabase 协作完成身份识别

    使用示例：
      recognizer = FaceRecognizer(device="cuda")
      emb = recognizer.get_embedding(crop)          # → (512,) ndarray
      sim = recognizer.compare(emb1, emb2)           # → 0.87
      name, score = recognizer.recognize(crop, db)   # → ("Byron", 0.85)
    """

    # 模型对应的推荐 det_size (更小的尺寸 = 更快推理)
    DEFAULT_DET_SIZES = {
        "buffalo_l": (96, 96),
        "buffalo_s": (320, 320),
        "buffalo_sc": (320, 320),
    }

    def __init__(
        self,
        model_name: str = "buffalo_s",
        device: Optional[str] = None,
        det_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        初始化识别器并加载 InsightFace 模型。

        Args:
            model_name: "buffalo_s"(快速) | "buffalo_l"(精准)
            device:     "cuda" / "cpu" / None(自动)
            det_size:   内部检测尺寸，None 时自动选择最佳值。
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface 未安装。请执行: pip install insightface>=0.7.3"
            )

        if det_size is None:
            det_size = self.DEFAULT_DET_SIZES.get(model_name, (112, 112))

        import onnxruntime as ort
        has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        if device is None:
            device = "cuda" if has_cuda else "cpu"
        if device == "cuda" and not has_cuda:
            device = "cpu"

        self.device = device
        ctx_id = 0 if device == "cuda" else -1

        print(f"[FaceRecognizer] 加载 InsightFace 模型: {model_name} "
              f"(device={self.device}, det_size={det_size}) ...")

        self.app = insightface.app.FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        self.threshold = 0.70
        print(f"[FaceRecognizer] 模型就绪 (threshold={self.threshold})")

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        从人脸裁剪图中提取 512 维归一化 embedding。

        内部流程：
          人脸 ROI → InsightFace 定位关键点 → 仿射对齐 → ArcFace 推理 → 512-d 向量

        Args:
            face_crop: BGR 格式的人脸裁剪图 (H, W, 3)，由 YOLO 检测框裁剪。

        Returns:
            np.ndarray: 512 维归一化 float32 向量，形状 (512,)。
            None: 无法提取（人脸太小、模糊、角度过大等）。

        注意：
          返回的 embedding 已做 L2 归一化，可直接用于余弦相似度计算。
        """
        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return None

        try:
            faces = self.app.get(face_crop)
            if len(faces) == 0:
                return None
            # embedding 已由 InsightFace 自动做 L2 归一化
            return faces[0].embedding
        except Exception:
            return None

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个 embedding 之间的余弦相似度。

        Args:
            emb1, emb2: 两个已归一化的 512 维向量。

        Returns:
            float: 余弦相似度 [-1.0, 1.0]，越接近 1.0 越可能是同一人。

        原理:
            因 embedding 已 L2 归一化，余弦相似度 = 向量点积。
        """
        return float(np.dot(emb1, emb2))

    def recognize(
        self,
        face_crop: np.ndarray,
        face_db: "FaceDatabase",
    ) -> Tuple[str, float]:
        """
        识别单张人脸裁剪图。

        完整流程：crop → embedding → 底库搜索 → (name, similarity)

        Args:
            face_crop: BGR 人脸裁剪图。
            face_db:   FaceDatabase 实例。

        Returns:
            (name, similarity): 识别结果。
            - 匹配成功: ("Byron", 0.85)
            - 未匹配:   ("Unknown", 0.00)
        """
        embedding = self.get_embedding(face_crop)
        if embedding is None:
            return ("Unknown", 0.0)

        result = face_db.search(embedding, threshold=self.threshold)
        if result is None:
            return ("Unknown", 0.0)

        return result
