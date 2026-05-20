"""
NCNN ArcFace face recognizer for ARM edge deployment (Raspberry Pi 5, etc.).

Uses ``ncnn`` (Python bindings) for inference.  Pre- and post-processing
logic mirrors the ONNX version.

Same interface as ``ArcFaceONNXRecognizer`` so the rest of the pipeline can
swap backends without code changes.

Usage::

    from core.recognition.arcface_ncnn import ArcFaceNCNNRecognizer

    recognizer = ArcFaceNCNNRecognizer(
        param_path="models/w600k_mbf.param",
        bin_path="models/w600k_mbf.bin",
    )
    embedding = recognizer.extract(aligned_face_112x112)
    # embedding: (512,) float32 L2-normalized

Engineering features (mirrors ``ArcFaceONNXRecognizer``):
    - Per-stage latency tracking (preprocess / inference / postprocess)
    - Deque-based rolling average with configurable window
    - ``avg_latency_ms`` property
    - ``close()`` for explicit session teardown
    - Lazy import of ``ncnn`` — ImportError only at construction time
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INPUT_NAME = "input.1"
"""Standard input blob name for ArcFace NCNN models."""


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
# Recognizer
# ---------------------------------------------------------------------------


class ArcFaceNCNNRecognizer:
    """
    NCNN-based ArcFace face recognizer.

    Parameters
    ----------
    param_path:
        Path to NCNN ``.param`` file.
    bin_path:
        Path to NCNN ``.bin`` file.
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
        num_threads: int = 4,
        stat_window: int = 30,
    ) -> None:
        self.param_path = param_path
        self.bin_path = bin_path
        self.num_threads = num_threads
        self._stat_window = stat_window

        # --- Validate model paths ---
        if not os.path.isfile(param_path):
            raise FileNotFoundError(
                f"ArcFace NCNN param file not found at: {param_path}\n"
                "Run scripts/convert_all_models.py to generate NCNN models.\n"
                "Expected output: models/w600k_mbf.param + models/w600k_mbf.bin"
            )
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(
                f"ArcFace NCNN bin file not found at: {bin_path}\n"
                "Run scripts/convert_all_models.py to generate NCNN models.\n"
                "Expected output: models/w600k_mbf.param + models/w600k_mbf.bin"
            )

        # --- Lazy-import ncnn (raises at construction time) ---
        ncnn_mod = _import_ncnn()

        # --- Build NCNN net ---
        self._net = ncnn_mod.Net()
        self._net.opt.use_vulkan_compute = False
        self._net.opt.num_threads = self.num_threads
        self._net.opt.use_packing_layout = True
        self._net.opt.use_fp16_storage = True
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
            "[ArcFaceNCNN] loaded %s + %s | threads=%d",
            os.path.basename(param_path),
            os.path.basename(bin_path),
            num_threads,
        )

        # --- Discover output blob name ---
        # Parse .param to find the single output blob
        self._output_name = self._discover_output_name(param_path)

        # --- Latency tracking (deque-based rolling average) ---
        self._pre_times: deque = deque(maxlen=stat_window)
        self._inf_times: deque = deque(maxlen=stat_window)
        self._post_times: deque = deque(maxlen=stat_window)
        self._total_times: deque = deque(maxlen=stat_window)

    # ------------------------------------------------------------------
    # .param file parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_output_name(param_path: str) -> str:
        """Parse NCNN .param file to find the single output blob name.

        ArcFace has a single output.  We find blob names that appear as
        layer outputs but NOT as inputs to any other layer.
        """
        all_outputs: list = []
        all_inputs: list = []

        with open(param_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                input_count = int(parts[1])
                output_count = int(parts[2])
            except (ValueError, IndexError):
                continue

            idx = 3
            for _ in range(input_count):
                if idx < len(parts):
                    all_inputs.append(parts[idx])
                    idx += 1
            for _ in range(output_count):
                if idx < len(parts):
                    all_outputs.append(parts[idx])
                    idx += 1

        input_set = set(all_inputs)
        graph_outputs = [n for n in all_outputs if n not in input_set]

        if graph_outputs:
            return graph_outputs[0]

        # Fallback: last output in the file
        if all_outputs:
            logger.warning(
                "[ArcFaceNCNN] could not determine graph output from .param, "
                "using last output: %s",
                all_outputs[-1],
            )
            return all_outputs[-1]

        logger.warning(
            "[ArcFaceNCNN] could not determine output from .param, "
            "falling back to default output name 'fc1'"
        )
        return "fc1"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(aligned_face: np.ndarray):
        """BGR uint8 112x112 aligned face -> NCNN Mat.

        ArcFace normalisation: ``(val - 127.5) / 127.5``
        (Note: SCRFD uses ``(val - 127.5) / 128.0`` — ArcFace is different.)
        """
        ncnn_mod = _import_ncnn()

        # BGR -> RGB (model expects RGB input)
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Convert to NCNN Mat (RGB, HWC -> CHW internally)
        mat = ncnn_mod.Mat.from_pixels(
            rgb,
            ncnn_mod.Mat.PixelType.PIXEL_RGB,
            112,
            112,
        )

        # ArcFace: mean 127.5, norm 1/127.5
        mean_vals = [127.5, 127.5, 127.5]
        norm_vals = [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5]
        mat.substract_mean_normalize(mean_vals, norm_vals)

        return mat

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(output: np.ndarray) -> np.ndarray:
        """Convert raw NCNN output to L2-normalised (512,) embedding."""
        embedding = output.astype(np.float32).flatten()
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
        if self._net is None:
            raise RuntimeError("ArcFaceNCNNRecognizer net has been closed")
        t0 = time.perf_counter()

        # --- Preprocess ---
        blob = self._preprocess(aligned_face)
        t1 = time.perf_counter()

        # --- Inference ---
        ncnn_mod = _import_ncnn()
        extractor = self._net.create_extractor()
        extractor.input(_INPUT_NAME, blob)
        out_mat = extractor.extract(self._output_name)
        if out_mat is None:
            raise RuntimeError(
                f"Failed to extract NCNN blob: {self._output_name}"
            )
        raw_output = np.array(ncnn_mod.to_numpy(out_mat), dtype=np.float32).flatten()
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

    def warmup(self) -> None:
        """Run a dummy inference to warm up the NCNN model.

        Creates a zero-filled ``(112, 112, 3)`` aligned face and runs
        ``extract()`` on it.
        """
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        self.extract(dummy)
        logger.info("[ArcFaceNCNN] warmup complete")

    def close(self) -> None:
        """Release the NCNN net.

        Further calls to ``extract()`` or ``warmup()`` will raise an error.
        Subsequent calls to ``close()`` are no-ops.
        """
        if self._net is not None:
            self._net = None  # type: ignore[assignment]
            logger.info("[ArcFaceNCNN] net released")

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
