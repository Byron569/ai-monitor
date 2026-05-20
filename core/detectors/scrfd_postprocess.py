"""
Pure NumPy implementation of SCRFD postprocessing chain.

Provides bbox decoding, NMS, and landmark decoding for SCRFD model outputs.
Reference formulas from insightface, zero insightface dependency.

Usage:
    from core.detectors.scrfd_postprocess import (
        generate_anchor_centers,
        decode_boxes,
        nms,
        decode_landmarks,
        scrfd_postprocess,
    )
"""

from typing import List, Optional, Tuple

import numpy as np
from utils.geometry import compute_iou


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------


def generate_anchor_centers(
    input_size: int,
    strides: List[int],
    num_anchors: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate anchor center coordinates and stride values for all stride levels.

    Each stride level produces ``(input_size / stride)^2 * num_anchors`` anchors
    whose centers are placed at the centre of each cell in the downsampled feature
    map, mapped back to the original image space.  SCRFD uses 2 anchors per cell.

    Args:
        input_size: Input image size (assumed square, e.g. 640).
        strides: List of stride values, e.g. ``[8, 16, 32]``.
        num_anchors: Anchors per spatial cell (default 2 for SCRFD).

    Returns:
        anchor_centers: ``(N, 2)`` float32 array of ``(cx, cy)`` per anchor.
        anchor_strides: ``(N,)`` float32 array of stride value per anchor.
    """
    all_centers: List[np.ndarray] = []
    all_strides: List[np.ndarray] = []

    for stride in strides:
        fmap_size = input_size // stride
        cx = np.arange(0, fmap_size, dtype=np.float32) * stride + stride // 2
        cy = np.arange(0, fmap_size, dtype=np.float32) * stride + stride // 2
        cxv, cyv = np.meshgrid(cx, cy)
        centers = np.stack([cxv.ravel(), cyv.ravel()], axis=1)
        # Repeat for num_anchors per cell (SCRFD uses 2)
        centers = np.repeat(centers, num_anchors, axis=0)
        all_centers.append(centers)
        all_strides.append(np.full(len(centers), stride, dtype=np.float32))

    return np.concatenate(all_centers, axis=0), np.concatenate(all_strides, axis=0)


# ---------------------------------------------------------------------------
# Box decoding
# ---------------------------------------------------------------------------


def decode_boxes(
    raw_boxes: np.ndarray,
    anchor_centers: np.ndarray,
    anchor_strides: np.ndarray,
) -> np.ndarray:
    """Decode raw SCRFD box outputs to ``(x1, y1, x2, y2)`` format.

    SCRFD uses *distance-based* encoding: the four values are
    ``(left, top, right, bottom)`` offsets from the anchor centre,
    normalised by the stride::

        x1 = anchor_cx - left  * stride
        y1 = anchor_cy - top   * stride
        x2 = anchor_cx + right * stride
        y2 = anchor_cy + bottom * stride

    Args:
        raw_boxes: ``(N, 4)`` array of raw ``(left, top, right, bottom)``.
        anchor_centers: ``(N, 2)`` array of ``(cx, cy)`` anchor centers.
        anchor_strides: ``(N,)`` array of stride for each anchor.

    Returns:
        ``(N, 4)`` array of decoded boxes in ``(x1, y1, x2, y2)`` format.
    """
    left   = raw_boxes[:, 0] * anchor_strides
    top    = raw_boxes[:, 1] * anchor_strides
    right  = raw_boxes[:, 2] * anchor_strides
    bottom = raw_boxes[:, 3] * anchor_strides
    x1 = anchor_centers[:, 0] - left
    y1 = anchor_centers[:, 1] - top
    x2 = anchor_centers[:, 0] + right
    y2 = anchor_centers[:, 1] + bottom
    return np.stack([x1, y1, x2, y2], axis=1)


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4,
) -> List[int]:
    """Greedy IoU-based non-maximum suppression.

    Sorts detections by score descending, greedily selects the highest-scoring
    box, then suppresses any remaining box whose IoU with the selected box
    exceeds ``iou_threshold``.

    Uses ``utils.geometry.compute_iou`` for pairwise IoU calculation.

    Args:
        boxes: ``(N, 4)`` array of ``(x1, y1, x2, y2)`` boxes.
        scores: ``(N,)`` array of confidence scores.
        iou_threshold: IoU threshold for suppression (default 0.4).

    Returns:
        List of indices of boxes to keep (sorted by descending score).
    """
    if len(boxes) == 0:
        return []

    # descending score order
    order = np.argsort(-scores)
    keep: List[int] = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]

        # compute iou between box[i] and every remaining box
        ious = np.array(
            [compute_iou(tuple(boxes[i]), tuple(boxes[j])) for j in rest],
            dtype=np.float64,
        )
        order = rest[ious <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# Landmark decoding
# ---------------------------------------------------------------------------


def decode_landmarks(
    raw_kps: np.ndarray,
    anchor_centers: np.ndarray,
    anchor_strides: np.ndarray,
) -> np.ndarray:
    """Decode raw SCRFD landmark outputs.

    Decoding formula for each of the 5 facial landmarks::

        lx = anchor_cx + raw_lx * stride
        ly = anchor_cy + raw_ly * stride

    Args:
        raw_kps: ``(N, 10)`` array of raw landmark predictions
            ``(lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lx5, ly5)``.
        anchor_centers: ``(N, 2)`` array of ``(cx, cy)`` anchor centers.
        anchor_strides: ``(N,)`` array of stride for each anchor.

    Returns:
        ``(N, 5, 2)`` array of decoded landmark coordinates.
    """
    N = raw_kps.shape[0]
    landmarks = np.zeros((N, 5, 2), dtype=raw_kps.dtype)

    for k in range(5):
        landmarks[:, k, 0] = (
            anchor_centers[:, 0] + raw_kps[:, 2 * k] * anchor_strides
        )
        landmarks[:, k, 1] = (
            anchor_centers[:, 1] + raw_kps[:, 2 * k + 1] * anchor_strides
        )

    return landmarks


# ---------------------------------------------------------------------------
# Full pipeline convenience
# ---------------------------------------------------------------------------


def scrfd_postprocess(
    raw_scores_per_stride: List[np.ndarray],
    raw_boxes_per_stride: List[np.ndarray],
    raw_kps_per_stride: List[np.ndarray],
    anchor_centers_list: List[np.ndarray],
    anchor_strides_list: List[np.ndarray],
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> List[Tuple[int, int, int, int, float, List[Tuple[int, int]]]]:
    """Run full SCRFD postprocessing pipeline across all stride levels.

    Pipeline:
        1. For each stride level: threshold scores, decode boxes, decode kps.
        2. Concatenate candidates from all levels.
        3. Greedy NMS.
        4. Return in standard format.

    The caller is responsible for splitting raw model outputs per stride level
    and generating the corresponding anchor centers/ strides via
    :func:`generate_anchor_centers`.

    If you have a single flat output (all anchors merged), pre-split them
    before calling this function.

    Args:
        raw_scores_per_stride: List of ``(num_anchors, 1)`` raw score arrays,
            one per stride level.
        raw_boxes_per_stride: List of ``(num_anchors, 4)`` raw box arrays,
            one per stride level.
        raw_kps_per_stride: List of ``(num_anchors, 10)`` raw landmark arrays,
            one per stride level.
        anchor_centers_list: List of ``(num_anchors, 2)`` anchor center arrays,
            one per stride level.
        anchor_strides_list: List of ``(num_anchors,)`` stride arrays,
            one per stride level.
        conf_threshold: Score threshold (default 0.5).
        nms_threshold: IoU threshold for NMS (default 0.4).

    Returns:
        List of detections in standard format::

            [(x1, y1, x2, y2, confidence, landmarks), ...]

        where ``landmarks`` is
        ``[(lx, ly), (rx, ry), (nx, ny), (lmx, lmy), (rmx, rmy)]``.
    """
    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_landmarks: List[np.ndarray] = []

    for stride_idx in range(len(raw_scores_per_stride)):
        scores = raw_scores_per_stride[stride_idx].ravel()
        boxes_raw = raw_boxes_per_stride[stride_idx]
        kps_raw = raw_kps_per_stride[stride_idx]
        ac = anchor_centers_list[stride_idx]
        as_ = anchor_strides_list[stride_idx]

        # filter by confidence
        mask = scores >= conf_threshold
        if not np.any(mask):
            continue

        scores_f = scores[mask]
        boxes_raw_f = boxes_raw[mask]
        kps_raw_f = kps_raw[mask]
        ac_f = ac[mask]
        as_f = as_[mask]

        boxes_decoded = decode_boxes(boxes_raw_f, ac_f, as_f)
        kps_decoded = decode_landmarks(kps_raw_f, ac_f, as_f)

        all_boxes.append(boxes_decoded)
        all_scores.append(scores_f)
        all_landmarks.append(kps_decoded)

    if not all_boxes:
        return []

    boxes_all = np.concatenate(all_boxes, axis=0)
    scores_all = np.concatenate(all_scores, axis=0)
    landmarks_all = np.concatenate(all_landmarks, axis=0)

    keep = nms(boxes_all, scores_all, nms_threshold)

    results = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_all[idx]
        conf = float(scores_all[idx])
        kps = landmarks_all[idx]
        landmarks = [(int(kps[k, 0]), int(kps[k, 1])) for k in range(5)]
        results.append((int(x1), int(y1), int(x2), int(y2), conf, landmarks))

    return results
