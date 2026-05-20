"""
Pure ONNX Runtime SCRFD face detector.

Replaces ``SCRFDDetector`` (which uses insightface) with a zero-dependency
ONNX Runtime implementation.  Only requires ``onnxruntime`` and ``numpy``.

Interface:
    detector = SCRFDONNXDetector(model_path="models/scrfd_500m.onnx")
    detections = detector.detect(frame)
    # detections: List[(x1, y1, x2, y2, confidence, landmarks)]
    # landmarks: [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]

Engineering features (preserved from ``SCRFDDetector``):
    - Per-stage latency tracking (preprocess / inference / postprocess)
    - Deque-based rolling average with configurable window
    - ``stats_report()`` returning formatted string
    - ``verbose`` parameter for periodic logging
    - ``avg_latency_ms`` property
    - ``close()`` for explicit session teardown
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from core.detectors.scrfd_postprocess import (
    generate_anchor_centers,
    scrfd_postprocess,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INPUT_NAME = "input.1"
"""Standard input tensor name for SCRFD ONNX models."""

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SCRFDONNXDetector:
    """
    Pure ONNX Runtime SCRFD face detector.

    Parameters
    ----------
    model_path:
        Path to ``.onnx`` model file (e.g. ``models/scrfd_500m.onnx``).
    input_size:
        Detection input size (square), default ``640``.
    conf_threshold:
        Confidence threshold, default ``0.5``.
    nms_threshold:
        NMS IoU threshold, default ``0.4``.
    device:
        ``"cuda"`` or ``"cpu"``.  When ``"cpu"``, ARM-friendly session options
        are applied (4 intra-op threads, full graph optimisation, etc.).
    verbose:
        Log per-frame latency statistics every ``stat_window`` frames.
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
        input_size: int = 640,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cuda",
        verbose: bool = False,
        stat_window: int = 30,
        box_padding: float = 0.15,
        box_shift_x: float = 0.0,
        box_shift_y: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.verbose = verbose
        self._stat_window = stat_window
        self.box_padding = box_padding
        self.box_shift_x = box_shift_x
        self.box_shift_y = box_shift_y

        # --- Validate model path ---
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SCRFD ONNX model not found at: {model_path}\n"
                "Download the model from the insightface model zoo:\n"
                "  https://github.com/deepinsight/insightface/tree/master/model_zoo\n"
                "or copy it from your local insightface cache:\n"
                "  ~/.insightface/models/buffalo_l/det_500m.onnx"
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
        logger.info(
            "[SCRFDONNX] loaded %s | input=%d | device=%s | conf=%.2f",
            model_path,
            input_size,
            self.device,
            conf_threshold,
        )

        # --- Cache output names for output extraction ---
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Build lookup: for each stride find the output index of score/bbox/kps
        self._scores_indices: List[int] = []
        self._bboxes_indices: List[int] = []
        self._kps_indices: List[int] = []
        self._detected_strides: List[int] = []

        self._parse_outputs_by_name()

        # Flag: when name-based matching failed, use shape-based grouping
        self._use_shape_grouping = len(self._scores_indices) == 0

        # Pre-generate anchor centres / strides for each detection head
        self._anchor_centers: List[np.ndarray] = []
        self._anchor_strides: List[np.ndarray] = []
        detected = self._detected_strides if self._detected_strides else [8, 16, 32]
        for stride in detected:
            ac, as_ = generate_anchor_centers(self.input_size, [stride])
            self._anchor_centers.append(ac)
            self._anchor_strides.append(as_)

        # --- Latency tracking (deque-based rolling average) ---
        self._frame_count = 0
        self._pre_times: deque = deque(maxlen=stat_window)
        self._inf_times: deque = deque(maxlen=stat_window)
        self._post_times: deque = deque(maxlen=stat_window)
        self._total_times: deque = deque(maxlen=stat_window)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_outputs_by_name(self) -> None:
        """Parse ONNX output names to group them by type and stride.

        Attempts to match outputs by semantic naming conventions:

        - ``score_8``, ``scores_8``, ``stride8_score``, etc.
        - ``bbox_8``, ``boxes_8``, ``stride8_bbox``, etc.
        - ``kps_8``, ``landmarks_8``, ``stride8_kps``, etc.

        If name-based matching fails, falls back to positional ordering
        assuming the conventional 9-output layout (3 scores, 3 bboxes,
        3 kps) with strides [8, 16, 32].
        """
        scores: List[Tuple[int, int]] = []   # (stride, output_index)
        bboxes: List[Tuple[int, int]] = []
        kps: List[Tuple[int, int]] = []

        for idx, name in enumerate(self._output_names):
            nl = name.lower()
            stride = self._extract_stride_from_name(nl)
            if stride is None:
                continue
            if any(t in nl for t in ("score", "scores", "cls", "confidence")):
                scores.append((stride, idx))
            elif any(t in nl for t in ("bbox", "boxes", "box", "loc", "deltas")):
                bboxes.append((stride, idx))
            elif any(t in nl for t in ("kps", "landmark", "lmk", "keypoint")):
                kps.append((stride, idx))

        # Validate: we need exactly 3 of each type
        if len(scores) == 3 and len(bboxes) == 3 and len(kps) == 3:
            scores.sort(key=lambda x: x[0])
            bboxes.sort(key=lambda x: x[0])
            kps.sort(key=lambda x: x[0])
            self._scores_indices = [i for _, i in scores]
            self._bboxes_indices = [i for _, i in bboxes]
            self._kps_indices = [i for _, i in kps]
            self._detected_strides = [s for s, _ in scores]
            logger.debug(
                "[SCRFDONNX] matched outputs by name: strides=%s",
                self._detected_strides,
            )
            return

        # Fallback: shape-based grouping at inference time
        if len(self._output_names) >= 9:
            logger.warning(
                "[SCRFDONNX] name-based output matching failed; "
                "using shape-based grouping at inference time."
            )
            self._detected_strides = [8, 16, 32]
            # Indices stay empty; _extract_outputs will call _group_by_stride
            return
        raise RuntimeError(
            f"Cannot parse SCRFD outputs: found {len(self._output_names)} "
            f"outputs. Expected 9 outputs (3 scores, 3 bboxes, 3 kps).\n"
            f"Output names: {self._output_names}"
        )

    @staticmethod
    def _extract_stride_from_name(name: str) -> Optional[int]:
        """Extract stride integer from an output name like ``score_8``."""
        # Common patterns: score_8, stride8_score, /8/score, etc.
        # Try suffix/prefix number patterns
        match = re.search(r"_(\d+)$", name)  # score_8, bbox_16
        if match:
            return int(match.group(1))
        match = re.search(r"_stride(\d+)", name)  # stride8_score
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)_\w+", name)  # 8_score
        if match:
            return int(match.group(1))
        return None

    # ------------------------------------------------------------------
    # Output tensor squashing
    # ------------------------------------------------------------------

    @staticmethod
    def _squash_output(arr: np.ndarray, expected_dim: int) -> np.ndarray:
        """Convert a raw ONNX output tensor to ``(N, expected_dim)``.

        Handles common SCRFD output layouts:

        - ``(1, N, D)``           -> ``(N, D)``
        - ``(1, D, H, W)``       -> ``(N, D)``  where ``N = H * W``
        - ``(1, D, N, 1)``       -> ``(N, D)``
        - ``(1, H, W, D)``       -> ``(N, D)``
        - ``(1, N, 2)`` scores   -> ``(N, 1)``  (face class only)
        - ``(1, 2, N, 1)`` scores -> ``(N, 1)``
        """
        arr = np.squeeze(arr)  # remove batch dim + size-1 dims

        if arr.ndim == 1:
            # Fully flattened — reshape directly
            return arr.reshape(-1, expected_dim)

        if arr.ndim == 2:
            # Two common cases: (N, D) or (D, N) — also (N, 2) for scores
            if arr.shape[1] == expected_dim:
                return arr
            if arr.shape[0] == expected_dim:
                return arr.T
            # Score-specific: (N, 2) or (2, N) -> take face class
            if expected_dim == 1 and arr.shape[1] == 2:
                return arr[:, 1:2]
            if expected_dim == 1 and arr.shape[0] == 2:
                return arr.T[:, 1:2]
            return arr.reshape(-1, expected_dim)

        if arr.ndim == 3:
            # Layouts: (D, H, W) or (H, W, D)
            if arr.shape[0] == expected_dim:
                return arr.reshape(expected_dim, -1).T
            if arr.shape[2] == expected_dim:
                return arr.reshape(-1, expected_dim)
            # Score-specific: (N, 2, 1) -> squeeze + take face class
            if arr.shape[1] == 2 and expected_dim == 1:
                return arr[:, 1:2, :].reshape(-1, 1)
            if arr.shape[1] == expected_dim and arr.shape[2] == 1:
                return arr.reshape(-1, expected_dim)
            # (2, H, W) scores
            if expected_dim == 1 and arr.shape[0] == 2:
                return arr[1:2].reshape(-1, 1)
            return arr.reshape(-1, expected_dim)

        # Anything else (4D+) — brute-force reshape
        return arr.reshape(-1, expected_dim)

    @staticmethod
    def _group_by_stride(
        outputs_list: List[np.ndarray],
        output_names: List[str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Group outputs by type and stride based on actual shapes.

        Inference-time fallback when name-based output matching fails.
        Each raw output is reshaped to 2D, classified by its feature
        dimension (1 or 2 = score, 4 = bbox, 10 = kps), then sorted by
        the number of anchors (N) descending -- stride 8 has the largest
        N, stride 32 the smallest.

        Parameters
        ----------
        outputs_list:
            Raw ONNX Runtime session outputs (9 tensors).
        output_names:
            Corresponding output names (for diagnostics).

        Returns
        -------
        Tuple of (scores_list, bboxes_list, kps_list) each containing
        3 arrays in descending stride order (8, 16, 32).
        """
        scores_raw: List[np.ndarray] = []
        boxes_raw: List[np.ndarray] = []
        kps_raw: List[np.ndarray] = []
        for name, arr in zip(output_names, outputs_list):
            # Determine channel count. For simple 2D tensors use last dim.
            # For higher-rank tensors, scan all dimensions for the one matching {1,2,4,10}.
            if arr.ndim == 2:
                ncols = arr.shape[-1]
            else:
                ncols = 0
                for d in arr.shape:
                    if d in (2, 4, 10):
                        ncols = d
                        break
                if ncols == 0:
                    ncols = arr.shape[-1]

            if ncols in (1, 2):
                scores_raw.append(SCRFDONNXDetector._squash_output(arr, 1))
            elif ncols == 4:
                boxes_raw.append(SCRFDONNXDetector._squash_output(arr, 4))
            elif ncols == 10:
                kps_raw.append(SCRFDONNXDetector._squash_output(arr, 10))
            else:
                logger.warning(
                    "[SCRFDONNX] unhandled output '%s' shape %s", name, arr.shape,
                )
        # Sort by N descending (stride 8 = most anchors, stride 32 = fewest)
        scores_raw.sort(key=lambda x: x.shape[0], reverse=True)
        boxes_raw.sort(key=lambda x: x.shape[0], reverse=True)
        kps_raw.sort(key=lambda x: x.shape[0], reverse=True)

        if not (len(scores_raw) == 3 and len(boxes_raw) == 3 and len(kps_raw) == 3):
            raise RuntimeError(
                f"Shape-based output grouping failed: "
                f"{len(scores_raw)} scores, {len(boxes_raw)} bboxes, "
                f"{len(kps_raw)} kps (expected 3 each)"
            )

        return scores_raw, boxes_raw, kps_raw

    def _extract_outputs(
        self, session_outputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Extract and reshape the 9 outputs into per-stride lists.

        Uses name-based index lookup by default.  When name-based matching
        failed in ``_parse_outputs_by_name``, falls back to shape-based
        grouping via ``_group_by_stride``.

        Returns
        -------
        Tuple of (scores_list, bboxes_list, kps_list) where each is a list
        of 3 arrays (one per stride level) in descending stride order
        (8 -> 16 -> 32).
        """
        if self._use_shape_grouping:
            return self._group_by_stride(session_outputs, self._output_names)

        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kps_list: List[np.ndarray] = []

        for i in range(len(self._detected_strides)):
            raw_score = session_outputs[self._scores_indices[i]]
            raw_bbox = session_outputs[self._bboxes_indices[i]]
            raw_kps = session_outputs[self._kps_indices[i]]

            scores_list.append(self._squash_output(raw_score, 1))
            bboxes_list.append(self._squash_output(raw_bbox, 4))
            kps_list.append(self._squash_output(raw_kps, 10))

        return scores_list, bboxes_list, kps_list

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # BGR -> RGB (model expects RGB input)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32, copy=False)
        blob = (blob - 127.5) / 128.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
        return blob

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, list]]:
        """
        Detect faces in a BGR frame.

        Parameters
        ----------
        frame:
            BGR uint8 image, shape ``(H, W, 3)``.

        Returns
        -------
        List of ``(x1, y1, x2, y2, confidence, landmarks)`` tuples sorted by
        descending confidence.
        """
        if self._session is None:
            raise RuntimeError("SCRFDONNXDetector session has been closed")
        t0 = time.perf_counter()

        # --- Preprocess (direct resize — model was trained this way) ---
        blob = self._preprocess(frame)
        t1 = time.perf_counter()

        # --- Inference ---
        raw_outputs = self._session.run(None, {_INPUT_NAME: blob})
        t2 = time.perf_counter()

        # --- Extract outputs ---
        scores_list, bboxes_list, kps_list = self._extract_outputs(raw_outputs)

        # --- Postprocess ---
        results = scrfd_postprocess(
            raw_scores_per_stride=scores_list,
            raw_boxes_per_stride=bboxes_list,
            raw_kps_per_stride=kps_list,
            anchor_centers_list=self._anchor_centers,
            anchor_strides_list=self._anchor_strides,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        t3 = time.perf_counter()

        # --- Map coordinates back to original frame (model sees square) ---
        h, w = frame.shape[:2]
        x_scale = w / self.input_size
        y_scale = h / self.input_size
        scaled = []
        for (x1, y1, x2, y2, conf, landmarks) in results:
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)
            lms = [(int(lx * x_scale), int(ly * y_scale)) for lx, ly in landmarks]
            if self.box_padding > 0:
                bw = (x2 - x1) * self.box_padding
                bh = (y2 - y1) * self.box_padding
                # box_shift is fraction of original box size (e.g. -0.15 = shift up 15%)
                sx = int((x2 - x1) * self.box_shift_x)
                sy = int((y2 - y1) * self.box_shift_y)
                x1 = max(0, int(x1 - bw + sx))
                y1 = max(0, int(y1 - bh + sy))
                x2 = min(w, int(x2 + bw + sx))
                y2 = min(h, int(y2 + bh + sy))
            scaled.append((x1, y1, x2, y2, conf, lms))
        results = scaled

        # --- Record latencies ---
        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        self._pre_times.append(pre_ms)
        self._inf_times.append(inf_ms)
        self._post_times.append(post_ms)
        self._total_times.append(total_ms)

        if self.verbose and self._frame_count % self._stat_window == 0:
            logger.info(
                "[SCRFDONNX] pre:%(pre).1fms | infer:%(inf).1fms | "
                "post:%(post).1fms | total:%(total).1fms",
                {
                    "pre": self._avg_ms(self._pre_times),
                    "inf": self._avg_ms(self._inf_times),
                    "post": self._avg_ms(self._post_times),
                    "total": self._avg_ms(self._total_times),
                },
            )

        self._frame_count += 1
        return results

    def warmup(self) -> None:
        """Run a dummy inference to warm up the model and CUDA kernels.

        Creates a zero-filled ``(input_size, input_size, 3)`` frame and runs
        ``detect()`` on it.
        """
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("[SCRFDONNX] warmup complete")

    def close(self) -> None:
        """Release the ONNX Runtime session.

        Further calls to ``detect()`` or ``warmup()`` will raise an error.
        Subsequent calls to ``close()`` are no-ops.
        """
        if self._session is not None:
            self._session = None  # type: ignore[assignment]
            logger.info("[SCRFDONNX] session released")

    # ------------------------------------------------------------------
    # Performance statistics
    # ------------------------------------------------------------------

    @property
    def avg_latency_ms(self) -> float:
        """Rolling average of total inference latency in milliseconds."""
        return self._avg_ms(self._total_times)

    def stats_report(self) -> str:
        """Return a formatted performance report string."""
        if self._frame_count == 0:
            return "[SCRFDONNX] no data"
        return (
            f"[SCRFDONNX] frames:{self._frame_count} | "
            f"pre:{self._avg_ms(self._pre_times):.1f}ms | "
            f"infer:{self._avg_ms(self._inf_times):.1f}ms | "
            f"post:{self._avg_ms(self._post_times):.1f}ms | "
            f"total:{self._avg_ms(self._total_times):.1f}ms | "
            f"device:{self.device}"
        )

    @staticmethod
    def _avg_ms(dq: deque) -> float:
        """Compute mean of a deque; returns 0.0 if empty."""
        if not dq:
            return 0.0
        return sum(dq) / len(dq)
