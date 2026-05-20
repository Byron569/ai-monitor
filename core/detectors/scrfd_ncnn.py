"""
NCNN SCRFD face detector for ARM edge deployment (Raspberry Pi 5, etc.).

Uses ``ncnn`` (Python bindings) for inference.  Pre- and post-processing
logic is reused from the ONNX version (``scrfd_postprocess``).

Same interface as ``SCRFDONNXDetector`` so the rest of the pipeline can
swap backends without code changes.

Usage::

    from core.detectors.scrfd_ncnn import SCRFDNCNNDetector

    detector = SCRFDNCNNDetector(
        param_path="models/scrfd_500m.param",
        bin_path="models/scrfd_500m.bin",
    )
    detections = detector.detect(frame)

Engineering features (mirrors ``SCRFDONNXDetector``):
    - Per-stage latency tracking (preprocess / inference / postprocess)
    - Deque-based rolling average with configurable window
    - ``avg_latency_ms`` property
    - ``close()`` for explicit session teardown
    - Lazy import of ``ncnn`` — ImportError only at construction time
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

from core.detectors.scrfd_postprocess import (
    generate_anchor_centers,
    scrfd_postprocess,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INPUT_NAME = "input.1"
"""Standard input blob name for SCRFD NCNN models."""


# ---------------------------------------------------------------------------
# NCNN helper
# ---------------------------------------------------------------------------

def _import_ncnn():
    """Lazy import of ncnn.  Raises ImportError with install instructions."""
    try:
        import ncnn
        return ncnn
    except ImportError:
        raise ImportError(
            "ncnn Python bindings not found.\n"
            "Install on Raspberry Pi 5:\n"
            "  sudo apt install libncnn-dev\n"
            "  pip install ncnn\n\n"
            "Or build from source:\n"
            "  https://github.com/Tencent/ncnn\n"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SCRFDNCNNDetector:
    """
    NCNN-based SCRFD face detector.

    Parameters
    ----------
    param_path:
        Path to NCNN ``.param`` file.
    bin_path:
        Path to NCNN ``.bin`` file.
    input_size:
        Detection input size (square), default ``320`` (edge-optimised).
    conf_threshold:
        Confidence threshold, default ``0.5``.
    nms_threshold:
        NMS IoU threshold, default ``0.4``.
    num_threads:
        Number of NCNN threads (ARM big.LITTLE friendly), default ``4``.
    stat_window:
        Rolling-average window size, default ``30``.

    Raises
    ------
    FileNotFoundError
        If ``param_path`` or ``bin_path`` does not exist.
    ImportError
        If the ``ncnn`` Python package is not installed (raised at
        construction time, not at import time).
    """

    def __init__(
        self,
        param_path: str,
        bin_path: str,
        input_size: int = 320,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        num_threads: int = 4,
        stat_window: int = 30,
    ) -> None:
        self.param_path = param_path
        self.bin_path = bin_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        self._stat_window = stat_window

        # --- Validate model paths ---
        if not os.path.isfile(param_path):
            raise FileNotFoundError(
                f"SCRFD NCNN param file not found at: {param_path}\n"
                "Run scripts/convert_all_models.py to generate NCNN models.\n"
                "Expected output: models/scrfd_500m.param + models/scrfd_500m.bin"
            )
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(
                f"SCRFD NCNN bin file not found at: {bin_path}\n"
                "Run scripts/convert_all_models.py to generate NCNN models.\n"
                "Expected output: models/scrfd_500m.param + models/scrfd_500m.bin"
            )

        # --- Lazy-import ncnn (raises at construction time) ---
        ncnn_mod = _import_ncnn()

        # --- Build NCNN net ---
        self._net = ncnn_mod.Net()
        self._net.opt.use_vulkan_compute = False  # CPU-only for Pi 5
        self._net.opt.num_threads = self.num_threads
        self._net.opt.use_packing_layout = True
        self._net.opt.use_fp16_storage = True  # FP16 on ARM NEON
        self._net.opt.use_fp16_arithmetic = True

        ret_param = self._net.load_param(param_path)
        if ret_param != 0:
            raise RuntimeError(
                f"Failed to load NCNN param: {param_path} (error code: {ret_param})"
            )
        ret_bin = self._net.load_bin(bin_path)
        if ret_bin != 0:
            raise RuntimeError(
                f"Failed to load NCNN bin: {bin_path} (error code: {ret_bin})"
            )

        logger.info(
            "[SCRFDNCNN] loaded %s + %s | input=%d | threads=%d | conf=%.2f",
            os.path.basename(param_path),
            os.path.basename(bin_path),
            input_size,
            num_threads,
            conf_threshold,
        )

        # --- Discover output blob names ---
        # NCNN does not provide a direct API for listing all blob names.
        # Instead we parse the .param file to find output layer names.
        self._output_names = self._parse_output_names_from_param(param_path)

        # Build lookup: for each stride find the output index of score/bbox/kps
        self._scores_indices: List[int] = []
        self._bboxes_indices: List[int] = []
        self._kps_indices: List[int] = []
        self._detected_strides: List[int] = []
        self._score_blob_names: List[str] = []
        self._bbox_blob_names: List[str] = []
        self._kps_blob_names: List[str] = []

        self._parse_outputs_by_name()

        # Pre-generate anchor centres / strides for each detection head
        self._anchor_centers: List[np.ndarray] = []
        self._anchor_strides: List[np.ndarray] = []
        for stride in self._detected_strides:
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
    # .param file parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output_names_from_param(param_path: str) -> List[str]:
        """Parse NCNN .param file to extract output layer names.

        NCNN .param files list layers with outputs in the format::

            LayerType  input_count  output_count  input_blobs  output_blobs

        We scan for lines where the output blob name does NOT appear as an
        input to any other layer — these are the graph outputs.
        """
        all_outputs: List[str] = []
        all_inputs: List[str] = []

        with open(param_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip header line (magic number + layer count)
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # Format: layer_type input_count output_count input_blobs... output_blobs...
            # Example: "Convolution 1 1 input.1 78"  (1 input, 1 output)
            try:
                input_count = int(parts[1])
                output_count = int(parts[2])
            except (ValueError, IndexError):
                continue

            if input_count == 0 and output_count == 0:
                continue

            # Input blobs: after layer type (index 0), input_count, output_count
            idx = 3
            for _ in range(input_count):
                if idx < len(parts):
                    all_inputs.append(parts[idx])
                    idx += 1
            # Output blobs
            for _ in range(output_count):
                if idx < len(parts):
                    all_outputs.append(parts[idx])
                    idx += 1

        # Graph outputs are outputs that are never used as inputs
        input_set = set(all_inputs)
        graph_outputs = []
        seen = set()
        for name in all_outputs:
            if name not in input_set and name not in seen:
                graph_outputs.append(name)
                seen.add(name)

        if not graph_outputs:
            # Fallback: use all unique outputs (may include intermediates)
            graph_outputs = list(dict.fromkeys(all_outputs))

        logger.debug(
            "[SCRFDNCNN] discovered %d output blobs from .param: %s",
            len(graph_outputs),
            graph_outputs,
        )
        return graph_outputs

    @staticmethod
    def _extract_stride_from_name(name: str) -> Optional[int]:
        """Extract stride integer from an output blob name like ``score_8``."""
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

    def _parse_outputs_by_name(self) -> None:
        """Parse NCNN output blob names to group by type and stride.

        Attempts to match by semantic naming conventions, same as
        ``SCRFDONNXDetector._parse_outputs_by_name``.
        """
        scores: List[Tuple[int, int, str]] = []   # (stride, output_index, blob_name)
        bboxes: List[Tuple[int, int, str]] = []
        kps: List[Tuple[int, int, str]] = []

        for idx, name in enumerate(self._output_names):
            nl = name.lower()
            stride = self._extract_stride_from_name(nl)
            if stride is None:
                continue
            if any(t in nl for t in ("score", "scores", "cls", "confidence")):
                scores.append((stride, idx, name))
            elif any(t in nl for t in ("bbox", "boxes", "box", "loc", "deltas")):
                bboxes.append((stride, idx, name))
            elif any(t in nl for t in ("kps", "landmark", "lmk", "keypoint")):
                kps.append((stride, idx, name))

        # Validate: we need exactly 3 of each type
        if len(scores) == 3 and len(bboxes) == 3 and len(kps) == 3:
            scores.sort(key=lambda x: x[0])
            bboxes.sort(key=lambda x: x[0])
            kps.sort(key=lambda x: x[0])
            self._scores_indices = [i for _, i, _ in scores]
            self._bboxes_indices = [i for _, i, _ in bboxes]
            self._kps_indices = [i for _, i, _ in kps]
            self._score_blob_names = [n for _, _, n in scores]
            self._bbox_blob_names = [n for _, _, n in bboxes]
            self._kps_blob_names = [n for _, _, n in kps]
            self._detected_strides = [s for s, _, _ in scores]
            logger.debug(
                "[SCRFDNCNN] matched outputs by name: strides=%s, blobs=%s",
                self._detected_strides,
                [n for _, _, n in scores + bboxes + kps],
            )
            return

        # Fallback: positional ordering (9 outputs: 3 scores, 3 bboxes, 3 kps)
        if len(self._output_names) >= 9:
            logger.warning(
                "[SCRFDNCNN] name-based output matching failed for %d blobs; "
                "falling back to positional ordering (first 9).",
                len(self._output_names),
            )
            self._scores_indices = [0, 1, 2]
            self._bboxes_indices = [3, 4, 5]
            self._kps_indices = [6, 7, 8]
            self._score_blob_names = self._output_names[0:3]
            self._bbox_blob_names = self._output_names[3:6]
            self._kps_blob_names = self._output_names[6:9]
            self._detected_strides = [8, 16, 32]
        else:
            raise RuntimeError(
                f"Cannot parse SCRFD NCNN outputs: found "
                f"{len(self._output_names)} output blobs.\n"
                f"Blob names: {self._output_names}\n"
                f"Expected 9 outputs (3 scores, 3 bboxes, 3 kps) "
                f"or semantically named outputs."
            )

    # ------------------------------------------------------------------
    # Output tensor handling
    # ------------------------------------------------------------------

    @staticmethod
    def _ncnn_mat_to_numpy(mat: "ncnn.Mat") -> np.ndarray:
        """Convert an NCNN Mat to a numpy array.

        NCNN Mat stores data in CHW layout.  We convert to a flat numpy
        array and reshape based on the dimensions.
        """
        ncnn_mod = _import_ncnn()
        dims = mat.dims
        c = mat.c
        h = mat.h
        w = mat.w

        if dims == 1:
            # 1D: (N,)
            return np.array(ncnn_mod.to_numpy(mat), dtype=np.float32).reshape(-1)

        if dims == 2:
            # 2D: (H, W) — NCNN stores as (w, h) in 2D
            return np.array(ncnn_mod.to_numpy(mat), dtype=np.float32).reshape(h, w)

        if dims == 3:
            # 3D: (C, H, W)
            arr = np.array(ncnn_mod.to_numpy(mat), dtype=np.float32)
            return arr.reshape(c, h, w)

        # dims == 0 or unknown
        return np.array(ncnn_mod.to_numpy(mat), dtype=np.float32).flatten()

    def _ncnn_extract_outputs(self, extractor) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Extract all 9 outputs from NCNN extractor.

        Returns (scores_list, bboxes_list, kps_list), each a list of 3 arrays
        in increasing stride order.
        """
        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kps_list: List[np.ndarray] = []

        for i in range(len(self._detected_strides)):
            # Extract score blob
            mat_score = extractor.extract(self._score_blob_names[i])
            if mat_score is None:
                raise RuntimeError(
                    f"Failed to extract NCNN blob: {self._score_blob_names[i]}"
                )
            arr_score = self._ncnn_mat_to_numpy(mat_score)
            # Squash to (N, 1)
            scores_list.append(self._squash_output(arr_score, 1))

            # Extract bbox blob
            mat_bbox = extractor.extract(self._bbox_blob_names[i])
            arr_bbox = self._ncnn_mat_to_numpy(mat_bbox)
            bboxes_list.append(self._squash_output(arr_bbox, 4))

            # Extract kps blob
            mat_kps = extractor.extract(self._kps_blob_names[i])
            arr_kps = self._ncnn_mat_to_numpy(mat_kps)
            kps_list.append(self._squash_output(arr_kps, 10))

        return scores_list, bboxes_list, kps_list

    @staticmethod
    def _squash_output(arr: np.ndarray, expected_dim: int) -> np.ndarray:
        """Convert a flattened/raw NCNN output to ``(N, expected_dim)``.

        Mirrors ``SCRFDONNXDetector._squash_output`` but handles the
        simpler NCNN output layouts (NCNN tends to produce flatter outputs).
        """
        # Handle 1D fully flattened
        if arr.ndim == 1:
            return arr.reshape(-1, expected_dim)

        # Handle 2D directly
        if arr.ndim == 2:
            if arr.shape[1] == expected_dim:
                return arr
            if arr.shape[0] == expected_dim:
                return arr.T
            return arr.reshape(-1, expected_dim)

        # Handle 3D: (C, H, W) typical for NCNN
        if arr.ndim == 3:
            c, h, w = arr.shape
            if c == expected_dim:
                return arr.reshape(expected_dim, -1).T
            if h == expected_dim:
                return arr.reshape(h, -1).T
            if w == expected_dim:
                return arr.reshape(-1, expected_dim)
            return arr.reshape(-1, expected_dim)

        # Anything else
        return arr.reshape(-1, expected_dim)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray):
        """BGR uint8 frame -> NCNN Mat.

        Processing pipeline:
            1. BGR -> RGB (model expects RGB input)
            2. Resize to ``(input_size, input_size)``
            3. Convert to NCNN Mat via ``from_pixels``
            4. Apply mean subtraction (127.5) and normalisation (1/128.0)

        Returns
        -------
        ncnn.Mat
            Preprocessed blob ready for ``extractor.input()``.
        """
        ncnn_mod = _import_ncnn()

        # BGR -> RGB (model expects RGB input)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert to NCNN Mat (RGB, HWC layout)
        # NCNN's from_pixels handles HWC->CHW conversion internally
        mat = ncnn_mod.Mat.from_pixels(
            resized,
            ncnn_mod.Mat.PixelType.PIXEL_RGB,
            self.input_size,
            self.input_size,
        )

        # Normalise: subtract mean, divide by std
        mean_vals = [127.5, 127.5, 127.5]
        norm_vals = [1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0]
        mat.substract_mean_normalize(mean_vals, norm_vals)

        return mat

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
        if self._net is None:
            raise RuntimeError("SCRFDNCNNDetector net has been closed")
        t0 = time.perf_counter()

        # --- Preprocess ---
        blob = self._preprocess(frame)
        t1 = time.perf_counter()

        # --- Inference ---
        ncnn_mod = _import_ncnn()
        extractor = self._net.create_extractor()
        extractor.input(_INPUT_NAME, blob)

        # Extract all output blobs
        scores_list, bboxes_list, kps_list = self._ncnn_extract_outputs(extractor)
        t2 = time.perf_counter()

        # --- Postprocess ---
        raw_results = scrfd_postprocess(
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
        results = []
        for (x1, y1, x2, y2, conf, landmarks) in raw_results:
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)
            lms = [(int(lx * x_scale), int(ly * y_scale)) for lx, ly in landmarks]
            results.append((x1, y1, x2, y2, conf, lms))

        # --- Record latencies ---
        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        self._pre_times.append(pre_ms)
        self._inf_times.append(inf_ms)
        self._post_times.append(post_ms)
        self._total_times.append(total_ms)

        self._frame_count += 1
        return results

    def warmup(self) -> None:
        """Run a dummy inference to warm up the NCNN model.

        Creates a zero-filled ``(input_size, input_size, 3)`` frame and runs
        ``detect()`` on it.
        """
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("[SCRFDNCNN] warmup complete")

    def close(self) -> None:
        """Release the NCNN net.

        Further calls to ``detect()`` or ``warmup()`` will raise an error.
        Subsequent calls to ``close()`` are no-ops.
        """
        if self._net is not None:
            self._net = None  # type: ignore[assignment]
            logger.info("[SCRFDNCNN] net released")

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
