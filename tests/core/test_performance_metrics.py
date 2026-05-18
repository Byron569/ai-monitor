"""
test_performance_metrics.py — 性能指标验收

Q1: 核心模块延迟是否在边缘设备可接受范围？
Q2: 多人场景是否仍保持低延迟？
Q3: SCRFD 降频是否生效？
"""

import sys, os, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from collections import deque
from core.track_memory import TrackMemory
from utils.motion_gate import MotionGate
from core.frame_scheduler import FrameScheduler

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def bench(func, *args, iterations=1000, **kwargs):
    times = deque(maxlen=iterations)
    for _ in range(iterations):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    return avg, max(times)


def test_tracker_latency():
    print("\n--- 场景A: Tracker 延迟 ---")
    tm = TrackMemory()
    avg, peak = bench(tm.match_hungarian, [(100, 100, 200, 200)], iterations=500)
    check(f"匈牙利匹配 < 2ms (avg={avg:.2f}ms peak={peak:.2f}ms)", avg < 2.0)
    avg2, peak2 = bench(tm.update, {1: (100, 100, 200, 200)}, iterations=500)
    check(f"TrackMemory update < 0.5ms (avg={avg2:.2f}ms)", avg2 < 0.5)


def test_motion_latency():
    print("\n--- 场景B: Motion Gate 延迟 ---")
    gate = MotionGate(threshold=2.0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    avg, peak = bench(gate.check, frame, iterations=500)
    check(f"Motion Gate < 1ms (avg={avg:.2f}ms)", avg < 1.0)


def test_multi_person():
    print("\n--- 场景C: 多人场景 ---")
    tm = TrackMemory()
    dets = [(100, 100, 200, 200), (300, 100, 400, 200), (500, 100, 600, 200)]
    avg, peak = bench(tm.match_hungarian, dets, iterations=300)
    check(f"3人匹配 < 5ms (avg={avg:.2f}ms)", avg < 5.0)


def test_scheduler():
    print("\n--- 场景D: Scheduler 自适应 ---")
    sched = FrameScheduler(detection_interval=2, force_interval=15)
    det_count = 0
    for i in range(180):  # 6 seconds at 30fps
        should, reason = sched.should_detect(i, True, motion_score=5)
        if should:
            det_count += 1
    check(f"检测降频生效 (180帧→{det_count}次检测)", det_count < 180)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [PERFORMANCE TEST] Edge AI Benchmark")
    print("=" * 50)
    test_tracker_latency()
    test_motion_latency()
    test_multi_person()
    test_scheduler()
    print(f"\n  perf_ok={PASS} fail={FAIL}")
    return PASS, FAIL
