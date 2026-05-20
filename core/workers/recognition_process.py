"""
RecognitionProcess — multiprocess face recognition worker.

Extracts face embeddings from frames submitted via SharedMemory.
Subclass of ``BaseInferenceProcess`` — runs ArcFace ONNX inference in a
dedicated child process.

Lifecycle (child process):
    init_model() -> _warmup() -> cmd_loop (read cmd_queue, dispatch) -> _child_cleanup()

Communication protocol (same as BaseInferenceProcess):
    Commands (main -> child):  {"cmd": "submit", "slot_idx": N, "meta": {...}}
    Results (child -> main):   {"tid": int, "identity": str, "embedding": list, "similarity": float}

Meta fields expected:
    tid         — track ID (int)
    bbox        — face bounding box [x1, y1, x2, y2] (list of ints)
    landmarks   — optional face landmarks (list of (x, y) pairs, length >= 5)
    frame_h     — input frame height (int, default 480)
    frame_w     — input frame width (int, default 640)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.workers.base_process import BaseInferenceProcess

logger = logging.getLogger(__name__)

# ArcFace alignment reference landmarks (same as pipeline.py)
_ARCFACE_SRC = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


class RecognitionProcess(BaseInferenceProcess):
    """Multiprocess face recognition worker.

    Loads ``ArcFaceONNXRecognizer`` in the child process, reads frames from
    the SharedMemory ring buffer, extracts aligned face crops, and computes
    512-d L2-normalised embeddings.

    Parameters
    ----------
    model_path:
        Path to the ArcFace ONNX model file (e.g. ``models/w600k_mbf.onnx``).
    device:
        ``"cuda"`` or ``"cpu"``.
    stat_window:
        Rolling-average window size for latency tracking in the recognizer.
    name:
        Process name (used for logging and SHM naming).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        stat_window: int = 30,
        name: str = "recognition",
    ) -> None:
        super().__init__(name=name)
        self._model_path = model_path
        self._device = device
        self._stat_window = stat_window

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def init_model(self) -> None:
        """Load ArcFaceONNXRecognizer in the child process."""
        from core.recognition.arcface_onnx import ArcFaceONNXRecognizer

        self._recognizer = ArcFaceONNXRecognizer(
            model_path=self._model_path,
            device=self._device,
            stat_window=self._stat_window,
        )

    def _warmup(self) -> None:
        """Run one dummy inference to warm up the model."""
        self._recognizer.warmup()
        logger.info("[RecognitionProcess] warmup complete")

    def process_frame(self, slot_idx: int, meta: dict) -> dict[str, Any]:
        """Read a frame from SHM, extract aligned face, compute embedding.

        Args:
            slot_idx: Ring-buffer slot index.
            meta: Must contain ``"tid"`` and ``"bbox"``.
                  Optionally contains ``"landmarks"``, ``"frame_h"``,
                  ``"frame_w"``.

        Returns:
            Dict with keys ``tid``, ``identity``, ``embedding``, ``similarity``.
        """
        import cv2

        tid = meta.get("tid")
        bbox = meta.get("bbox")
        landmarks = meta.get("landmarks")
        frame_h = meta.get("frame_h", 480)
        frame_w = meta.get("frame_w", 640)

        if tid is None or bbox is None:
            logger.warning(
                "[RecognitionProcess] Missing tid or bbox in meta"
            )
            return {"tid": tid, "identity": "Unknown", "embedding": [], "similarity": 0.0}

        # -- Read full frame from SharedMemory --
        offset = slot_idx * self.SLOT_SIZE
        nbytes = frame_h * frame_w * 3
        frame_data = np.frombuffer(
            self._input_shm.buf,
            dtype=np.uint8,
            count=nbytes,
            offset=offset,
        ).reshape(frame_h, frame_w, 3).copy()

        # -- Extract and align face --
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)

        crop: np.ndarray | None = None

        if landmarks is not None and len(landmarks) >= 5:
            aligned = self._align_face(frame_data, np.array(landmarks), size=112)
            if aligned is not None and aligned.size > 0:
                crop = aligned

        if crop is None and x2 > x1 and y2 > y1:
            crop = frame_data[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, (112, 112))

        if crop is None or crop.size == 0:
            return {
                "tid": tid,
                "identity": "Unknown",
                "embedding": [],
                "similarity": 0.0,
            }

        # -- Compute embedding --
        embedding = self._recognizer.extract(crop)

        return {
            "tid": tid,
            "identity": "Unknown",  # DB search performed in pipeline harvest
            "embedding": embedding.tolist(),
            "similarity": 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_face(
        frame: np.ndarray,
        landmarks: np.ndarray,
        size: int = 112,
    ) -> np.ndarray | None:
        """Align a face using similarity transform and ArcFace landmarks.

        Args:
            frame: Input frame (H, W, 3).
            landmarks: 5 facial landmarks (5, 2).
            size: Output face size (default 112).

        Returns:
            Aligned face crop ``(size, size, 3)`` or ``None`` on failure.
        """
        import cv2

        dst = _ARCFACE_SRC * (size / 112.0)
        M = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst)
        if M is None or M[0] is None:
            return None
        return cv2.warpAffine(frame, M[0], (size, size), borderValue=0.0)
