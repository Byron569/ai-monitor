"""
test_recognition_scheduler.py — RecognitionScheduler 优先级调度验收

Q1: 新 track 优先识别？
Q2: 冷却期内不重复提交？
Q3: pending 状态跳过？
Q4: 已识别跳过 priority 2？
Q5: force_recognize 立即生效？
Q6: cleanup 清理过期状态？
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

from core.scheduler.recognition_scheduler import RecognitionScheduler

PASS, FAIL = 0, 0


class _MockPerson:
    def __init__(self, track_id, frame_seen=5, identified=False):
        self.track_id = track_id
        self.frame_seen = frame_seen
        self.is_identified = identified
        self.identity = "Byron" if identified else "Unknown"

    def __repr__(self):
        return f"MockPerson(id={self.track_id}, ident={self.is_identified})"


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def test_new_track_priority():
    print("\n--- 优先级1: 新 track 优先 ---")
    s = RecognitionScheduler(cooldown=300)
    p1 = _MockPerson(1, frame_seen=5)
    p2 = _MockPerson(2, frame_seen=1)  # too new, frame_seen < 3
    target = s.get_next([p1, p2], frame_id=0)
    check("新track=1被调度", target is not None and target.track_id == 1,
          f"实际={target.track_id if target else 'None'}")
    check("track=2太新被跳过(seen<3)", target.track_id != 2,
          f"实际={target.track_id if target else 'None'}")


def test_cooldown():
    print("\n--- 冷却期逻辑 ---")
    s = RecognitionScheduler(cooldown=300)
    p = _MockPerson(1, frame_seen=5)
    target = s.get_next([p], frame_id=0)
    check("第1次调度成功", target is not None)
    s.mark_submitted(1)
    s.mark_completed(1, frame_id=0)
    target2 = s.get_next([p], frame_id=50)
    check("冷却期内不重复调度(50<300)", target2 is None)
    target3 = s.get_next([p], frame_id=400)
    check("冷却期满可重新调度(400>=390=300+90backoff)", target3 is not None and target3.track_id == 1)


def test_pending_block():
    print("\n--- pending 状态阻塞 ---")
    s = RecognitionScheduler(cooldown=300)
    p = _MockPerson(1, frame_seen=5)
    target = s.get_next([p], frame_id=0)
    check("第1次调度成功", target is not None)
    s.mark_submitted(1)
    target2 = s.get_next([p], frame_id=300)
    check("pending中不可再次调度", target2 is None)


def test_mark_identified():
    print("\n--- 已识别跳过 priority 2 ---")
    s = RecognitionScheduler(cooldown=300)
    s.mark_submitted(1)
    s.mark_identified(1, frame_id=0)
    p1 = _MockPerson(1, frame_seen=5, identified=True)
    p2 = _MockPerson(2, frame_seen=5, identified=False)
    target = s.get_next([p1, p2], frame_id=0)
    check("已识别p1跳过，调度p2", target is not None and target.track_id == 2)


def test_mark_completed_fail_count():
    print("\n--- fail_count 递增 ---")
    s = RecognitionScheduler(cooldown=300)
    p = _MockPerson(1, frame_seen=5)
    s.mark_submitted(1)
    s.mark_completed(1, frame_id=0)
    s.mark_completed(1, frame_id=1)
    st = s._states.get(1)
    check("fail_count=2", st is not None and st.fail_count == 2)


def test_force_recognize():
    print("\n--- force_recognize 清理状态 ---")
    s = RecognitionScheduler(cooldown=300)
    p = _MockPerson(1, frame_seen=5)
    s.mark_submitted(1)
    s.mark_identified(1, frame_id=0)
    check("已识别状态存在", 1 in s._states)
    s.force_recognize(1)
    check("force后状态清除", 1 not in s._states)
    target = s.get_next([p], frame_id=0)
    check("force后立即可调度", target is not None and target.track_id == 1)


def test_cleanup_stale():
    print("\n--- cleanup 清理过期状态 ---")
    s = RecognitionScheduler(cooldown=300)
    p = _MockPerson(1, frame_seen=5)
    s.mark_completed(1, frame_id=0)
    s.mark_completed(2, frame_id=600)
    s.cleanup(active_ids=set(), frame_id=2500)
    check("tid=1 清理(2500-0>2000)", 1 not in s._states)
    check("tid=2 未清理(2500-600=1900<2000)", 2 in s._states)


def test_pending_count():
    print("\n--- pending_count 统计 ---")
    s = RecognitionScheduler(cooldown=300)
    check("初始pending=0", s.pending_count == 0)
    s.mark_submitted(1)
    s.mark_submitted(2)
    check("标记2个后pending=2", s.pending_count == 2)
    s.mark_completed(1, frame_id=0)
    check("完成1个后pending=1", s.pending_count == 1)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [SCHEDULER TEST] RecognitionScheduler")
    print("=" * 50)
    test_new_track_priority()
    test_cooldown()
    test_pending_block()
    test_mark_identified()
    test_mark_completed_fail_count()
    test_force_recognize()
    test_cleanup_stale()
    test_pending_count()
    print(f"\n  scheduler_ok={PASS} fail={FAIL}")
    return PASS, FAIL
