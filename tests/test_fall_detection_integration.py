"""摔倒检测集成测试 — 不依赖摄像头, 使用纯模型推理。"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fall_engine_imports():
    """验证 fall_engine 模块可正常导入。"""
    from plugins.fall_engine import set_config
    from plugins.fall_engine.config import FALL, FEATURES
    from plugins.fall_engine.backends import create_backend
    from plugins.fall_engine.fall_logic import evaluate_fall, calculate_angle
    from plugins.fall_engine.features import compute_all_features, yolo_to_5keypoints


def test_evaluate_fall_no_fall():
    """测试站立姿态不被误判为摔倒。"""
    from plugins.fall_engine import set_config
    from plugins.fall_engine.fall_logic import evaluate_fall

    set_config({
        "fall": {
            "horizontal_ar_threshold": 0.6, "angle_threshold": 120,
            "torso_inclination_threshold": 65, "min_fall_pose_duration": 1.0,
            "window_size": 10, "window_trigger_ratio": 0.5,
            "min_consecutive_triggers": 1, "min_standing_ar": 1.2,
            "re_threshold": 20, "gf_threshold": 8000, "head_descent_threshold": 0.2,
            "ema_alpha": 0.3, "sg_window": 7, "sg_polyorder": 2,
            "min_bbox_area": 3000, "rebound_threshold": 0.3, "rebound_frames": 2,
            "recovery_frames": 3,
        }
    })

    # 构造站立 person 数据 (AR = h/w ≈ 3.0, > 1.2)
    kp_5 = {"H": (320, 100), "N": (320, 200), "B": (320, 350),
            "KL": (300, 430), "KR": (340, 430)}
    person_data = {
        "pid": 1, "kp_5": kp_5, "aspect_ratio": 3.0,
        "bbox": [250, 50, 390, 480],
        "fall_state": {
            "is_potential_fall": False,
            "fall_start_time": None,
            "fall_detected": False,
            "trigger_history": [],
        },
        "history": [], "keypoints": None, "confs": None,
        "angle_keypoints": {
            "shoulder": (320, 200), "hip": (320, 350), "knee": (320, 430)
        },
    }

    result = evaluate_fall(person_data, time.time())
    assert result["fall_detected"] == False, f"站立不应被判为摔倒, got {result}"


def test_fall_detection_task_disabled():
    """验证 enabled=False 时 FallDetectionTask 不运行。"""
    from plugins.fall_detection import FallDetectionTask

    task = FallDetectionTask({"enabled": False})
    assert task.enabled == False
    assert task.should_run(0, [(1, [0, 0, 100, 200], "Unknown")], {}) == False


def test_fall_detection_task_runs():
    """验证 enabled=True 时 should_run 在正确时机返回 True。"""
    from plugins.fall_detection import FallDetectionTask

    task = FallDetectionTask({
        "enabled": True, "interval": 5,
        "model_path": "models/yolov8n-pose.onnx", "device": "cpu",
        "backend": "onnx", "confidence_threshold": 0.5,
    })
    if not task.enabled:
        print("SKIP: 无模型文件，跳过推理测试")
        return

    tracks = [(1, [100, 50, 300, 450], "Unknown")]
    # 异步 Worker 架构: should_run 永远返回 True, Worker 内部控制帧间隔
    assert task.should_run(0, tracks, {}) == True
    assert task.should_run(6, tracks, {}) == True


def test_evaluate_fall_detected():
    """测试模拟前倒过程 → 确认 FALL 被触发。"""
    from plugins.fall_engine import set_config
    from plugins.fall_engine.fall_logic import evaluate_fall

    set_config({
        "fall": {
            "horizontal_ar_threshold": 0.6, "angle_threshold": 120,
            "torso_inclination_threshold": 65, "min_fall_pose_duration": 0.01,
            "window_size": 10, "window_trigger_ratio": 0.5,
            "min_consecutive_triggers": 1, "min_standing_ar": 1.2,
            "re_threshold": 20, "gf_threshold": 8000, "head_descent_threshold": 0.2,
            "ema_alpha": 0.3, "sg_window": 7, "sg_polyorder": 2,
            "min_bbox_area": 3000, "rebound_threshold": 0.3, "rebound_frames": 2,
            "recovery_frames": 3,
        }
    })

    fall_state = {
        "is_potential_fall": False, "fall_start_time": None,
        "fall_detected": False, "trigger_history": [],
        "consecutive_triggers": 0, "trigger_gap_count": 0,
    }

    # 模拟: 站立 2 帧 → 倒地 20 帧 → 确认 FALL
    stand_kp = {"H": (320, 120), "N": (320, 200), "B": (320, 350),
                "KL": (300, 450), "KR": (340, 450)}
    fall_kp = {"H": (200, 400), "N": (280, 390), "B": (360, 380),
               "KL": (200, 430), "KR": (450, 430)}

    now = time.time()

    # Phase 1: 站立 (2 frames), AR = h/w ≈ 3.0
    for i in range(2):
        person_data = {
            "pid": 1, "kp_5": stand_kp, "aspect_ratio": 3.0,
            "bbox": [250, 50, 390, 480],
            "fall_state": fall_state, "history": [],
            "keypoints": None, "confs": None,
            "angle_keypoints": {"shoulder": (320, 200), "hip": (320, 350), "knee": (320, 450)},
        }
        result = evaluate_fall(person_data, now + i * 0.1)
    assert result["fall_detected"] == False, "站立不应触发"
    assert result["state"] == "Normal", f"应为 Normal, got {result['state']}"

    # Phase 2: 持续倒地 (20 frames)
    for i in range(20):
        person_data = {
            "pid": 1, "kp_5": fall_kp, "aspect_ratio": 0.3,
            "bbox": [180, 370, 470, 480],
            "fall_state": fall_state, "history": [],
            "keypoints": None, "confs": None,
            "angle_keypoints": {"shoulder": (280, 390), "hip": (360, 380), "knee": (200, 430)},
        }
        result = evaluate_fall(person_data, now + (i + 2) * 0.1)

    assert result["fall_detected"] == True, \
        f"持续倒地应触发 FALL, got state={result['state']} fall_detected={result['fall_detected']}"
    assert result["state"] == "FALL", f"应为 FALL, got {result['state']}"


if __name__ == "__main__":
    print("--- test_fall_engine_imports ---")
    test_fall_engine_imports()
    print("PASS")
    print("--- test_evaluate_fall_no_fall ---")
    test_evaluate_fall_no_fall()
    print("PASS")
    print("--- test_fall_detection_task_disabled ---")
    test_fall_detection_task_disabled()
    print("PASS")
    print("--- test_fall_detection_task_runs ---")
    test_fall_detection_task_runs()
    print("PASS")
    print("--- test_evaluate_fall_detected ---")
    test_evaluate_fall_detected()
    print("PASS")
    print("All tests passed!")
