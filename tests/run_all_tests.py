"""
run_all_tests.py — 统一测试入口

运行: python tests/run_all_tests.py
输出: tests/test_report.txt
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ORT_LOG_LEVEL"] = "3"

from tests.core import (
    test_motion_gate,
    test_tracking_system,
    test_behavior_system,
    test_region_events,
    test_performance_metrics,
    test_failure_recovery,
    test_recognition_scheduler,
    test_person_manager,
    test_track_reassociation,
    test_recognition_worker,
    test_metrics_logger,
    test_face_db,
    test_face_quality,
    test_recognition_worker_queue,
    test_embedding_cache,
    test_gallery_search,
    test_vision_task_interface,
    test_fall_detection_stub,
    test_iou_tracker,
    test_tracker_factory,
)

TESTS = [
    ("Motion Gate", test_motion_gate.run),
    ("Tracking System", test_tracking_system.run),
    ("Behavior Engine", test_behavior_system.run),
    ("Region Events", test_region_events.run),
    ("Performance", test_performance_metrics.run),
    ("Failure Recovery", test_failure_recovery.run),
    ("Recognition Scheduler", test_recognition_scheduler.run),
    ("Person Manager", test_person_manager.run),
    ("Track Reassociation", test_track_reassociation.run),
    ("Recognition Worker", test_recognition_worker.run),
    ("Metrics Logger", test_metrics_logger.run),
    ("Face Database", test_face_db.run),
    ("Face Quality", test_face_quality.run),
    ("Worker Queue", test_recognition_worker_queue.run),
    ("Embedding Cache", test_embedding_cache.run),
    ("Gallery Search", test_gallery_search.run),
    ("VisionTask Interface", test_vision_task_interface.run),
    ("FallDetection Stub", test_fall_detection_stub.run),
    ("IoU Tracker", test_iou_tracker.run),
    ("Tracker Factory", test_tracker_factory.run),
]


def main():
    t0 = time.time()
    results = {}
    total_pass = total_fail = 0

    for name, runner in TESTS:
        try:
            p, f = runner()
            results[name] = (p, f, "PASS" if f == 0 else "FAIL")
            total_pass += p
            total_fail += f
        except Exception as e:
            results[name] = (0, 1, f"ERROR: {e}")
            total_fail += 1

    elapsed = time.time() - t0

    # 输出汇总
    print("\n" + "=" * 60)
    print(" SYSTEM VALIDATION REPORT")
    print("=" * 60)
    for name, (p, f, status) in results.items():
        print(f"  {name}: {status} ({p} pass, {f} fail)")

    overall = "EDGE AI READY" if total_fail == 0 else "NEEDS FIX"
    print(f"\n  Overall: {overall}")
    print(f"  Total: {total_pass} pass, {total_fail} fail in {elapsed:.1f}s")
    print("=" * 60)

    # 写报告
    report = []
    report.append("=" * 60)
    report.append(" SYSTEM VALIDATION REPORT")
    report.append(f" {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    for name, (p, f, status) in results.items():
        report.append(f"  {name}: {status} ({p} pass, {f} fail)")
    report.append(f"\n  Overall: {overall}")
    report.append("=" * 60)

    report_path = os.path.join(os.path.dirname(__file__), "test_report.txt")
    with open(report_path, "w") as fout:
        fout.write("\n".join(report))
    print(f"\n  Report saved: {report_path}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
