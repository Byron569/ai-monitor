"""
统一后处理模块
将不同后端的原始输出转换为 list[Detection]（纯对象）
"""

import numpy as np
from plugins.fall_engine.detection import Detection, Keypoint


def ultralytics_results_to_detections(results) -> list:
    """
    将 ultralytics Results 对象转换为 list[Detection]

    Returns:
        list[Detection]: 统一检测对象列表
    """
    detections = []

    if not hasattr(results, "boxes") or results.boxes is None:
        return detections

    raw_xyxy = results.boxes.xyxy
    raw_conf = results.boxes.conf
    if hasattr(raw_xyxy, 'cpu'):
        raw_xyxy = raw_xyxy.cpu().numpy()
    if hasattr(raw_conf, 'cpu'):
        raw_conf = raw_conf.cpu().numpy()
    bboxes = raw_xyxy if isinstance(raw_xyxy, np.ndarray) else np.array(raw_xyxy)
    scores = raw_conf if isinstance(raw_conf, np.ndarray) else np.array(raw_conf)

    if bboxes.size == 0 or bboxes.ndim < 2:
        return detections
    if len(bboxes) == 0:
        return detections

    track_ids = [None] * len(bboxes)
    if hasattr(results.boxes, "id") and results.boxes.id is not None:
        raw_id = results.boxes.id
        if hasattr(raw_id, 'cpu'):
            track_ids = raw_id.cpu().numpy().astype(int).tolist()
        else:
            track_ids = np.array(raw_id).astype(int).tolist()

    for i in range(len(bboxes)):
        kps = []
        if (results.keypoints is not None and
                results.keypoints.xy is not None and
                results.keypoints.conf is not None and
                i < len(results.keypoints.xy)):
            kp_xy_arr = results.keypoints.xy[i]
            kp_conf_arr = results.keypoints.conf[i]
            if hasattr(kp_xy_arr, 'cpu'):
                kp_xy_arr = kp_xy_arr.cpu().numpy()
            if hasattr(kp_conf_arr, 'cpu'):
                kp_conf_arr = kp_conf_arr.cpu().numpy()
            for (x, y), c in zip(kp_xy_arr, kp_conf_arr):
                kps.append(Keypoint(x=float(x), y=float(y), confidence=float(c)))

        detections.append(Detection(
            bbox=[float(v) for v in bboxes[i]],
            score=float(scores[i]),
            class_id=0,
            keypoints=kps,
            track_id=int(track_ids[i]) if track_ids[i] is not None else None,
        ))

    return detections


def onnx_outputs_to_detections(outputs, frame_shape, conf_threshold=0.35, iou_threshold=0.45, input_size=640) -> list:
    """
    将 ONNX YOLOv8-Pose 输出转换为 list[Detection]

    ONNX 模型输出 cx,cy,w,h 是相对于 input_size x input_size 的像素坐标。
    需先归一化到 [0,1] 再映射回原始帧尺寸。
    """
    if isinstance(outputs, (list, tuple)):
        output = outputs[0]
    else:
        output = outputs

    if output.ndim == 3:
        output = output[0]
    if output.shape[0] < 5:
        return []
    if output.shape[0] < output.shape[1]:
        output = output.T

    frame_h, frame_w = frame_shape[:2]
    obj_conf = output[:, 4]
    mask = obj_conf > conf_threshold
    if not np.any(mask):
        return []

    cx, cy = output[mask, 0], output[mask, 1]
    w, h = output[mask, 2], output[mask, 3]
    obj_conf = obj_conf[mask]
    output = output[mask]

    x1 = (cx - w / 2) / input_size * frame_w
    y1 = (cy - h / 2) / input_size * frame_h
    x2 = (cx + w / 2) / input_size * frame_w
    y2 = (cy + h / 2) / input_size * frame_h

    # NMS
    areas = (x2 - x1) * (y2 - y1)
    order = obj_conf.argsort()[::-1]
    keep = []
    indices = np.arange(len(x1))
    while len(order) > 0:
        idx = order[0]
        keep.append(indices[idx])
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[order[1:]], x1[idx])
        yy1 = np.maximum(y1[order[1:]], y1[idx])
        xx2 = np.minimum(x2[order[1:]], x2[idx])
        yy2 = np.minimum(y2[order[1:]], y2[idx])
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        _iou = (w_inter * h_inter) / (areas[order[1:]] + areas[idx] - w_inter * h_inter + 1e-8)
        remaining = np.where(_iou <= iou_threshold)[0]
        order = order[remaining + 1]

    detections = []
    for idx in keep:
        kps = []
        try:
            kp_data = output[idx, 5:56]
            if len(kp_data) >= 51:
                for j in range(17):
                    kps.append(Keypoint(
                        x=float(kp_data[j * 3] / input_size * frame_w),
                        y=float(kp_data[j * 3 + 1] / input_size * frame_h),
                        confidence=float(1.0 / (1.0 + np.exp(-kp_data[j * 3 + 2]))),
                    ))
        except (IndexError, ValueError):
            pass

        detections.append(Detection(
            bbox=[float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
            score=float(obj_conf[idx]),
            class_id=0,
            keypoints=kps,
            track_id=None,
        ))

    return detections
