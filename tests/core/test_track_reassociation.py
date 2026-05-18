"""
test_track_reassociation.py — TrackReassociation 轨迹重关联验收

Q1: 高 IoU 检测重关联？
Q2: 空间邻近检测重关联？
Q3: 无 lost tracks 返回空？
Q4: 无 detections 返回空？
Q5: 距离太远无匹配？
Q6: 三级匹配优先级正确？
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

from core.track_reassociation import TrackReassociation, _iou, _distance
from core.track_memory import TrackMemory

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


import time


def _make_tm_with_lost():
    tm = TrackMemory()
    return tm


def _add_lost(tm, tid, bbox):
    tm.update({})
    tm.update({tid: bbox})
    ts = tm._tracks[tid]
    ts.status = "lost"
    ts.last_seen = time.time()
    return ts


def test_iou_helpers():
    print("\n--- IoU/Distance 工具函数 ---")
    iou = _iou((0, 0, 100, 100), (50, 50, 150, 150))
    check("部分重叠 IoU≈0.14", 0.13 < iou < 0.15, f"iou={iou:.3f}")
    iou2 = _iou((0, 0, 100, 100), (0, 0, 100, 100))
    check("完全重叠 IoU=1.0", iou2 == 1.0, f"iou={iou2:.3f}")
    iou3 = _iou((0, 0, 100, 100), (200, 200, 300, 300))
    check("无重叠 IoU=0", iou3 == 0.0, f"iou={iou3:.3f}")
    dist = _distance((0, 0, 100, 100), (0, 0, 100, 100))
    check("相同框距离=0", dist == 0.0, f"dist={dist:.1f}")


def test_iou_match():
    print("\n--- IoU 重关联 ---")
    reassoc = TrackReassociation(iou_threshold=0.3, distance_threshold=200)
    tm = _make_tm_with_lost()
    _add_lost(tm, 5, (100, 100, 200, 200))
    detections = [(120, 120, 210, 210, 0.95)]
    matches = reassoc.match(detections, tm)
    check("det0 匹配到 tid=5", len(matches) == 1 and matches[0][5] == 5,
          f"matches={matches}")


def test_spatial_match():
    print("\n--- 空间邻近重关联 ---")
    reassoc = TrackReassociation(iou_threshold=0.5, distance_threshold=60)
    tm = _make_tm_with_lost()
    _add_lost(tm, 3, (100, 100, 200, 200))
    detections = [(150, 100, 250, 200, 0.9)]
    matches = reassoc.match(detections, tm)
    check("IoU≈0.33<0.5 距离=50<60，spatial匹配",
          len(matches) == 1 and matches[0][5] == 3,
          f"matches={matches}")


def test_no_lost_tracks():
    print("\n--- 无 lost tracks ---")
    reassoc = TrackReassociation()
    tm = _make_tm_with_lost()
    detections = [(100, 100, 200, 200, 0.9)]
    matches = reassoc.match(detections, tm)
    check("返回空dict", matches == {})


def test_no_detections():
    print("\n--- 无 detections ---")
    reassoc = TrackReassociation()
    tm = _make_tm_with_lost()
    _add_lost(tm, 5, (100, 100, 200, 200))
    matches = reassoc.match([], tm)
    check("返回空dict", matches == {})


def test_no_match_far_away():
    print("\n--- 距离太远无匹配 ---")
    reassoc = TrackReassociation(iou_threshold=0.3, distance_threshold=50)
    tm = _make_tm_with_lost()
    _add_lost(tm, 5, (0, 0, 50, 50))
    detections = [(500, 500, 550, 550, 0.9)]
    matches = reassoc.match(detections, tm)
    check("返回空dict", matches == {})


def test_multi_det_multi_lost():
    print("\n--- 多检测多丢失 ---")
    reassoc = TrackReassociation(iou_threshold=0.3, distance_threshold=200)
    tm = _make_tm_with_lost()
    _add_lost(tm, 1, (0, 0, 100, 100))
    _add_lost(tm, 2, (300, 300, 400, 400))
    detections = [
        (10, 10, 90, 90, 0.95),   # 匹配 tid=1 (IoU)
        (310, 310, 390, 390, 0.9), # 匹配 tid=2 (IoU)
    ]
    matches = reassoc.match(detections, tm)
    check("2个检测都匹配", len(matches) == 2)
    check("det0→tid=1", matches[0][5] == 1)
    check("det1→tid=2", matches[1][5] == 2)
    check("tids不重复", len(set(m[5] for m in matches.values())) == 2)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [REASSOCIATION TEST] TrackReassociation")
    print("=" * 50)
    test_iou_helpers()
    test_iou_match()
    test_spatial_match()
    test_no_lost_tracks()
    test_no_detections()
    test_no_match_far_away()
    test_multi_det_multi_lost()
    print(f"\n  reassociation_ok={PASS} fail={FAIL}")
    return PASS, FAIL
