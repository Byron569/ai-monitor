"""跨摄像头人物匹配。HSV 直方图 + 匈牙利算法。"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_color_histogram(img, bbox, nbins=8):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    upper_y2 = y1 + h // 2
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y1:upper_y2, x1:x2] = 1
    if mask.sum() == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [nbins, 2 * nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist


def match_cross_camera(tracked_a, tracked_b, threshold=0.5):
    if not tracked_a or not tracked_b:
        return []
    n_a, n_b = len(tracked_a), len(tracked_b)
    corr_matrix = np.zeros((n_a, n_b))
    for i in range(n_a):
        hist_a = tracked_a[i].get("hist")
        if hist_a is None:
            continue
        for j in range(n_b):
            hist_b = tracked_b[j].get("hist")
            if hist_b is None:
                continue
            corr_matrix[i][j] = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    cost_matrix = np.where(corr_matrix > 0, -corr_matrix, 1.0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched = []
    for r, c in zip(row_ind, col_ind):
        if corr_matrix[r][c] > threshold:
            matched.append((r, c))
    return matched


def merge_tracking_ids(tracked_a, tracked_b, matched_pairs, global_id_map, pid_to_global=None):
    if pid_to_global is None:
        pid_to_global = {}
        for (pa, pb), gid in global_id_map.items():
            pid_to_global[pa] = gid
            pid_to_global[pb] = gid
    next_gid = max(global_id_map.values(), default=0) + 1
    for a_idx, b_idx in matched_pairs:
        pid_a = tracked_a[a_idx]["pid"]
        pid_b = tracked_b[b_idx]["pid"]
        key = (pid_a, pid_b)
        if key not in global_id_map:
            existing = pid_to_global.get(pid_a) or pid_to_global.get(pid_b)
            if existing is not None:
                global_id_map[key] = existing
                pid_to_global[pid_a] = existing
                pid_to_global[pid_b] = existing
            else:
                global_id_map[key] = next_gid
                pid_to_global[pid_a] = next_gid
                pid_to_global[pid_b] = next_gid
                next_gid += 1
    return pid_to_global
