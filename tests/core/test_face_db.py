"""
test_face_db.py — FaceDatabase CRUD 验收

Q1: add_face + search 找到匹配？
Q2: 低于阈值返回 None？
Q3: remove_face 删除记录？
Q4: 空数据库 search 返回 None？
Q5: get_all_names 去重？
Q6: save/load 持久化往返？
Q7: 多 embedding 同名取最高相似度？
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from database.face_db import FaceDatabase

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def _make_emb(seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_add_search_match():
    print("\n--- 注册 + 搜索匹配 ---")
    db = FaceDatabase()
    emb = _make_emb(42)
    db.add_face("Byron", emb)
    check("count=1", db.count == 1)
    result = db.search(emb, threshold=0.4)
    check("匹配Byron", result is not None and result[0] == "Byron")
    check("相似度≈1.0", result is not None and result[1] > 0.99,
          f"sim={result[1]:.4f}" if result else "None")


def test_threshold_filter():
    print("\n--- 阈值过滤 ---")
    db = FaceDatabase()
    emb1 = _make_emb(42)
    db.add_face("Byron", emb1)
    emb2 = _make_emb(99)
    result = db.search(emb2, threshold=0.9)
    check("低于阈值返回None", result is None,
          f"result={result}")


def test_remove_face():
    print("\n--- 删除人脸 ---")
    db = FaceDatabase()
    db.add_face("Byron", _make_emb(42))
    db.add_face("Byron", _make_emb(99))
    db.add_face("Alice", _make_emb(7))
    check("初始3条", db.count == 3)
    removed = db.remove_face("Byron")
    check("删除2条", removed == 2)
    check("剩余1条", db.count == 1)
    check("剩余Alice", db.names == ["Alice"])


def test_empty_search():
    print("\n--- 空数据库搜索 ---")
    db = FaceDatabase()
    check("count=0", db.count == 0)
    result = db.search(_make_emb(42))
    check("返回None", result is None)


def test_get_all_names():
    print("\n--- 去重名字列表 ---")
    db = FaceDatabase()
    db.add_face("Byron", _make_emb(42))
    db.add_face("Byron", _make_emb(99))
    db.add_face("Alice", _make_emb(7))
    names = db.get_all_names()
    check("2个去重名字", len(names) == 2)
    check("包含Byron", "Byron" in names)
    check("包含Alice", "Alice" in names)
    check("排序", names[0] == "Alice")


def test_save_load_roundtrip():
    print("\n--- 持久化往返 ---")
    db1 = FaceDatabase()
    emb = _make_emb(42)
    db1.add_face("Byron", emb)
    db1.add_face("Alice", _make_emb(7))
    save_path = os.path.join(os.path.dirname(__file__), "_test_db.pkl")
    db1.save(save_path)
    db2 = FaceDatabase()
    db2.load(save_path)
    check("加载后count=2", db2.count == 2)
    check("名字一致", db2.names == db1.names)
    result = db2.search(emb, threshold=0.4)
    check("加载后仍可匹配", result is not None and result[0] == "Byron")
    os.remove(save_path)


def test_load_nonexistent():
    print("\n--- 加载不存在文件 ---")
    db = FaceDatabase()
    path = os.path.join(os.path.dirname(__file__), "_nonexistent.pkl")
    if os.path.exists(path):
        os.remove(path)
    db.load(path)
    check("count=0", db.count == 0)


def test_type_error():
    print("\n--- 类型检查 ---")
    db = FaceDatabase()
    try:
        db.add_face("Test", [1, 2, 3])
        check("TypeError被抛出", False)
    except TypeError:
        check("TypeError被抛出", True)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [FACE DB TEST] FaceDatabase")
    print("=" * 50)
    test_add_search_match()
    test_threshold_filter()
    test_remove_face()
    test_empty_search()
    test_get_all_names()
    test_save_load_roundtrip()
    test_load_nonexistent()
    test_type_error()
    print(f"\n  facedb_ok={PASS} fail={FAIL}")
    return PASS, FAIL
