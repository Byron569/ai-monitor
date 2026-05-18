"""
test_motion_gate.py — Motion Gate 验收测试

Q1: 静止画面是否不触发 SCRFD？
Q2: 轻微变化是否不误触发？
Q3: 真实运动是否正常触发？
"""

import sys, os, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from utils.motion_gate import MotionGate

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def test_static():
    print("\n--- 场景A: 静止画面 ---")
    gate = MotionGate(threshold=2.0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    activations = 0
    for _ in range(50):
        active, _ = gate.check(frame)
        if active:
            activations += 1
    check("motion=false 持续", activations == 0,
          f"静止画面触发 {activations} 次")
    check("scrfd_call≈0", activations == 0)


def test_light_change():
    print("\n--- 场景B: 轻微光照变化 ---")
    gate = MotionGate(threshold=2.0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    gate.check(frame)
    activations = 0
    for i in range(50):
        noise = np.random.randint(-3, 3, frame.shape, dtype=np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        active, _ = gate.check(noisy)
        if active:
            activations += 1
    check("不误触发", activations <= 2,
          f"误触发 {activations}/50 次（≤2=正常）")


def test_real_motion():
    print("\n--- 场景C: 真实运动 ---")
    gate = MotionGate(threshold=2.0)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    gate.check(frame)
    moving = np.roll(frame, 30, axis=1)
    gate.check(moving)
    activations = 0
    for _ in range(50):
        # 逐帧变化：随机位置添加白色方块模拟运动
        frame2 = moving.copy()
        offset = np.random.randint(0, 200)
        frame2[200:250, offset:offset + 50] = 255
        active, _ = gate.check(frame2)
        if active:
            activations += 1
    check("motion=true 触发", activations >= 15,
          f"仅触发 {activations}/50 次（≥15=正常）")


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [MOTION TEST] Motion Gate Validation")
    print("=" * 50)
    test_static()
    test_light_change()
    test_real_motion()
    print(f"\n  motion_trigger_ok={PASS} fail={FAIL}")
    return PASS, FAIL
