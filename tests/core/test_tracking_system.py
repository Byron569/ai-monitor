"""
test_tracking_system.py — Tracker 主导系统验收

Q1: 单人持续移动 track_id 是否稳定？
Q2: 两人交叉是否 ID switch 少？
Q3: 短暂遮挡 track_memory 是否恢复？
"""

import sys, os, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

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


def test_single_move():
    print("\n--- 场景A: 单人持续移动 ---")
    tm = TrackMemory(ttl_active=1.5, ttl_lost=1.5)
    switches = 0
    prev_tid = None
    for i in range(100):
        bbox = (100 + i, 100, 200 + i, 200)
        dets = [bbox]
        matches = tm.match_hungarian(dets)
        if matches:
            tid = list(matches.values())[0]
            if prev_tid is not None and tid != prev_tid:
                switches += 1
            prev_tid = tid
            tm.update({tid: bbox})
    check("track_id 稳定", switches == 0,
          f"ID switch {switches} 次（期望 0）")


def test_crossing():
    print("\n--- 场景B: 两人交叉运动 ---")
    tm = TrackMemory(ttl_active=1.5, ttl_lost=1.5)
    switches = 0
    prev = {}
    for i in range(60):
        if i < 30:
            a = (50 + i * 3, 100, 150 + i * 3, 200)
            b = (400 - i * 3, 100, 500 - i * 3, 200)
        else:
            a = (400 - (i - 30) * 3, 100, 500 - (i - 30) * 3, 200)
            b = (50 + (i - 30) * 3, 100, 150 + (i - 30) * 3, 200)
        matches = tm.match_hungarian([a, b])
        cur = {v: k for k, v in matches.items()}
        for tid, det_idx in cur.items():
            if tid in prev and prev[tid] != det_idx:
                switches += 1
        prev = cur
        for det_idx, tid in matches.items():
            tm.update({tid: [a, b][det_idx]})
    check("交叉后 ID switch 少", switches <= 4,
          f"ID switch {switches} 次（≤4=可接受）")


def test_occlusion():
    print("\n--- 场景C: 短暂遮挡恢复 ---")
    tm = TrackMemory(ttl_active=1.5, ttl_lost=1.5)
    # 首次手动创建 track
    dets = [(100, 100, 200, 200)]
    m = tm.match_hungarian(dets)
    if not m:
        tid = 1
        tm.update({tid: dets[0]})
    else:
        tid = list(m.values())[0]
    # 遮挡 20 帧
    for _ in range(20):
        tm.update({})
    dets2 = [(110, 110, 210, 210)]
    m2 = tm.match_hungarian(dets2)
    recovered = len(m2) > 0 and list(m2.values())[0] == tid
    check("遮挡后恢复", recovered,
          f"期望 tid={tid}, 实际={list(m2.values()) if m2 else 'None'}")
    # 长期遮挡 > ttl_lost → 过期
    for _ in range(60):
        tm.update({})
        time.sleep(0.03)  # 模拟真实帧间隔
    m3 = tm.match_hungarian(dets2)
    expired = len(m3) == 0
    if not expired and len(m3) > 0:
        expired = list(m3.values())[0] != tid
    check("超时后失效", expired, "长期遮挡后不应恢复")


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [TRACK TEST] Tracking System Validation")
    print("=" * 50)
    test_single_move()
    test_crossing()
    test_occlusion()
    print(f"\n  track_switch_ok={PASS} fail={FAIL}")
    return PASS, FAIL
