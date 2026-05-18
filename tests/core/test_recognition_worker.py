"""
test_recognition_worker.py — RecognitionWorker 异步识别验收

Q1: submit 正常入队？
Q2: 队列满时 submit 返回 False？
Q3: poll_results 收割结果？
Q4: stop 停止线程？
Q5: pending_count 准确？
"""

import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from core.workers.recognition_worker import RecognitionWorker

PASS, FAIL = 0, 0


class _MockRecognizer:
    def __init__(self):
        self.threshold = 0.7

    def get_embedding(self, crop):
        return np.random.randn(512).astype(np.float32)


class _MockDatabase:
    def search(self, embedding, threshold=0.7):
        return ("TestUser", 0.85)


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def test_submit_and_result():
    print("\n--- 提交任务 + 收割结果 ---")
    rec = _MockRecognizer()
    db = _MockDatabase()
    worker = RecognitionWorker(rec, db, max_queue_size=8)
    worker.start()
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok = worker.submit(1, crop)
    check("submit成功", ok)
    time.sleep(0.5)
    results = worker.poll_results()
    check("收到结果", len(results) > 0,
          f"实际={len(results)}")
    if results:
        tid, name, sim, emb, qs = results[0]
        check("track_id正确", tid == 1)
        check("name=TestUser", name == "TestUser")
    worker.stop()
    worker.join(timeout=1)


def test_full_queue_reject():
    print("\n--- 队列满时拒绝 ---")
    rec = _MockRecognizer()
    db = _MockDatabase()
    worker = RecognitionWorker(rec, db, max_queue_size=2)
    worker._running = True
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok1 = worker.submit(1, crop)
    ok2 = worker.submit(2, crop)
    check("前2个入队成功", ok1 and ok2)
    ok3 = worker.submit(3, crop)
    check("第3个被拒绝", not ok3)
    check("queue_size=2", worker.queue_size == 2)


def test_stop_and_running():
    print("\n--- stop 停止线程 ---")
    rec = _MockRecognizer()
    db = _MockDatabase()
    worker = RecognitionWorker(rec, db, max_queue_size=8)
    check("初始_running=True", worker._running)
    worker.stop()
    check("stop后_running=False", not worker._running)
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok = worker.submit(1, crop)
    check("stop后submit返回False", not ok)


def test_pending_count():
    print("\n--- pending_count ---")
    rec = _MockRecognizer()
    db = _MockDatabase()
    worker = RecognitionWorker(rec, db, max_queue_size=8)
    worker._running = True
    check("初始pending=0", worker.pending_count == 0)
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    worker.submit(1, crop)
    worker.submit(2, crop)
    check("2任务pending=2", worker.pending_count == 2)


def test_concurrent_results():
    print("\n--- 多任务并发结果 ---")
    rec = _MockRecognizer()
    db = _MockDatabase()
    worker = RecognitionWorker(rec, db, max_queue_size=16)
    worker.start()
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    for i in range(5):
        worker.submit(i, crop)
    time.sleep(1.0)
    results = worker.poll_results()
    check("收到5个结果", len(results) == 5,
          f"实际={len(results)}")
    tids = {r[0] for r in results}
    check("track_id不重复", len(tids) == 5)
    worker.stop()
    worker.join(timeout=1)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [WORKER TEST] RecognitionWorker")
    print("=" * 50)
    test_submit_and_result()
    test_full_queue_reject()
    test_stop_and_running()
    test_pending_count()
    test_concurrent_results()
    print(f"\n  worker_ok={PASS} fail={FAIL}")
    return PASS, FAIL
