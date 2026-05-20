"""
Pure ONNX Runtime ArcFace face recognizer.

Replaces the insightface-based ``FaceRecognizer`` with a zero-dependency
ONNX Runtime implementation.  Only requires ``onnxruntime`` and ``numpy``.

Interface::

    recognizer = ArcFaceONNXRecognizer(model_path="models/w600k_mbf.onnx")
    embedding = recognizer.extract(aligned_face_112x112)
    # embedding: (512,) float32 L2-normalized

Engineering features (mirrors ``SCRFDONNXDetector``):
    - Per-stage latency tracking (preprocess / inference / postprocess)
    - Deque-based rolling average with configurable window
    - ``avg_latency_ms`` property
    - ``close()`` for explicit session teardown
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_INPUT_NAME = "input.1"
"""Standard input tensor name for ArcFace ONNX models."""


class ArcFaceONNXRecognizer:
    """
    Pure ONNX Runtime ArcFace face recognizer.

    Parameters
    ----------
    model_path:
        Path to ``.onnx`` model file (e.g. ``models/w600k_mbf.onnx``).
    device:
        ``"cuda"`` or ``"cpu"``.  When ``"cpu"``, ARM-friendly session options
        are applied (4 intra-op threads, full graph optimisation, etc.).
    stat_window:
        Rolling-average window size, default ``30``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        stat_window: int = 30,
        threshold: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._stat_window = stat_window
        self.threshold = threshold

        # --- Validate model path ---
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at: {model_path}\n"
                "Download the model from the insightface model zoo:\n"
                "  https://github.com/deepinsight/insightface/tree/master/model_zoo\n"
                "or copy it from your local insightface cache:\n"
                "  ~/.insightface/models/buffalo_l/w600k_r50.onnx"
            )

        # --- Build session ---
        sess_options = ort.SessionOptions()
        if self.device == "cpu":
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
        else:
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        provider = (
            "CUDAExecutionProvider"
            if self.device == "cuda"
            else "CPUExecutionProvider"
        )
        self._session: ort.InferenceSession = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[provider],
        )

        # Cache output name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            "[ArcFaceONNX] loaded %s | device=%s",
            model_path,
            self.device,
        )

        # --- Latency tracking (deque-based rolling average) ---

        self._pre_times: deque = deque(maxlen=stat_window)
        self._inf_times: deque = deque(maxlen=stat_window)
        self._post_times: deque = deque(maxlen=stat_window)
        self._total_times: deque = deque(maxlen=stat_window)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(aligned_face: np.ndarray) -> np.ndarray:
        # BGR -> RGB (model expects RGB input)
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32, copy=False)
        blob = (blob - 127.5) / 127.5
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
        return blob

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(output: np.ndarray) -> np.ndarray:
        """Convert raw ONNX output to L2-normalised (512,) embedding."""
        embedding = np.squeeze(output).astype(np.float32)
        # Ensure (512,)
        if embedding.ndim != 1:
            embedding = embedding.flatten()
        # L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract a 512-d L2-normalised face embedding from an aligned face crop.

        Parameters
        ----------
        aligned_face:
            BGR uint8 aligned face, shape ``(112, 112, 3)``.

        Returns
        -------
        np.ndarray
            L2-normalised float32 embedding of shape ``(512,)``.
        """
        if self._session is None:
            raise RuntimeError("ArcFaceONNXRecognizer session has been closed")
        t0 = time.perf_counter()

        # --- Preprocess ---
        blob = self._preprocess(aligned_face)
        t1 = time.perf_counter()

        # --- Inference ---
        raw_output = self._session.run(
            [self._output_name], {_INPUT_NAME: blob}
        )[0]
        t2 = time.perf_counter()

        # --- Postprocess ---
        embedding = self._postprocess(raw_output)
        t3 = time.perf_counter()

        # --- Record latencies ---
        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        self._pre_times.append(pre_ms)
        self._inf_times.append(inf_ms)
        self._post_times.append(post_ms)
        self._total_times.append(total_ms)

        return embedding

    def get_embedding(self, aligned_face_112x112: np.ndarray) -> np.ndarray:
        """Alias for ``extract`` — backward compatibility with old FaceRecognizer API."""
        return self.extract(aligned_face_112x112)

    def warmup(self) -> None:
        """Run a dummy inference to warm up the model and CUDA kernels.

        Creates a zero-filled ``(112, 112, 3)`` aligned face and runs
        ``extract()`` on it.
        """
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        self.extract(dummy)
        logger.info("[ArcFaceONNX] warmup complete")

    def close(self) -> None:
        """Release the ONNX Runtime session.

        Further calls to ``extract()`` or ``warmup()`` will raise an error.
        Subsequent calls to ``close()`` are no-ops.
        """
        if self._session is not None:
            self._session = None  # type: ignore[assignment]
            logger.info("[ArcFaceONNX] session released")

    # ------------------------------------------------------------------
    # Performance statistics
    # ------------------------------------------------------------------

    @property
    def avg_latency_ms(self) -> float:
        """Rolling average of total inference latency in milliseconds."""
        return self._avg_ms(self._total_times)

    @staticmethod
    def _avg_ms(dq: deque) -> float:
        """Compute mean of a deque; returns 0.0 if empty."""
        if not dq:
            return 0.0
        return sum(dq) / len(dq)
