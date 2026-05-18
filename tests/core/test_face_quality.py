"""
test_face_quality.py — FaceQualityFilter 验收

Q1: 正常尺寸人脸通过？
Q2: 太小的人脸拒绝？
Q3: 越界bbox拒绝？
Q4: 异常长宽比拒绝？
Q5: 模糊人脸拒绝？
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from core.face_quality import FaceQualityFilter

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")

def _make_frame(h, w):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_valid_face():
    print("\n--- 正常人脸通过 ---")
    f = FaceQualityFilter(min_face_size=48, blur_threshold=10, min_quality_score=0.1)
    frame = _make_frame(480, 640)
    result = f.evaluate(frame, (200, 150, 300, 300))
    check("passed=True", result.passed, f"reason={result.reason}")
    check("score>0", result.score > 0, f"score={result.score:.2f}")


def test_too_small():
    print("\n--- 太小人脸拒绝 ---")
    f = FaceQualityFilter(min_face_size=48, blur_threshold=80, min_quality_score=0.55)
    frame = _make_frame(480, 640)
    result = f.evaluate(frame, (10, 10, 30, 40))
    check("passed=False", not result.passed)
    check("reason=too_small", "too_small" in result.reason)


def test_out_of_bounds():
    print("\n--- 越界bbox拒绝 ---")
    f = FaceQualityFilter(min_face_size=48)
    frame = _make_frame(480, 640)
    result = f.evaluate(frame, (-10, 10, 100, 100))
    check("passed=False (x1<0)", not result.passed)
    check("reason=out_of_bounds", result.reason == "out_of_bounds")


def test_bad_aspect():
    print("\n--- 异常长宽比拒绝 ---")
    f = FaceQualityFilter(min_face_size=48)
    frame = _make_frame(640, 640)
    result = f.evaluate(frame, (100, 100, 150, 350))
    check("passed=False (h/w>2.5)", not result.passed)
    check("reason=bad_aspect", "bad_aspect" in result.reason)


def test_blurry():
    print("\n--- 模糊人脸拒绝 ---")
    f = FaceQualityFilter(min_face_size=48, blur_threshold=500, min_quality_score=0.1)
    frame = _make_frame(480, 640)
    frame[:] = 128
    result = f.evaluate(frame, (200, 150, 300, 300))
    check("passed=False (blurry)", not result.passed, f"blur_score={result.blur_score:.1f}")
    check("blur_score≈0", result.blur_score < 10, f"blur_score={result.blur_score:.1f}")


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [QUALITY TEST] FaceQualityFilter")
    print("=" * 50)
    test_valid_face()
    test_too_small()
    test_out_of_bounds()
    test_bad_aspect()
    test_blurry()
    print(f"\n  quality_ok={PASS} fail={FAIL}")
    return PASS, FAIL
