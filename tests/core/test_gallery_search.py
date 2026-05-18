"""
test_gallery_search.py — FaceDatabase 向量化搜索验收

Q1: 向量化搜索匹配？
Q2: threshold过滤？
Q3: 移除后重建矩阵？
Q4: 空库返回None？
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["ORT_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)

import numpy as np
from database.face_db import FaceDatabase

PASS, FAIL = 0, 0

def _make_emb(seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name}  -- {detail}")

def test_vectorized_match():
    print("\n--- 向量化匹配 ---")
    db = FaceDatabase()
    for i in range(5):
        db.add_face(f"User{i}", _make_emb(i))
    q = _make_emb(0)
    result = db.search(q, threshold=0.4)
    check("匹配User0", result is not None and result[0] == "User0")
    check("相似度≈1.0", result is not None and result[1] > 0.99,
          f"sim={result[1]:.4f}" if result else "None")


def test_threshold_filter():
    print("\n--- 阈值过滤 ---")
    db = FaceDatabase()
    db.add_face("Byron", _make_emb(42))
    q = _make_emb(99)
    result = db.search(q, threshold=0.9)
    check("低相似度返回None", result is None)


def test_empty_db():
    print("\n--- 空库搜索 ---")
    db = FaceDatabase()
    result = db.search(_make_emb(0))
    check("返回None", result is None)


def test_matrix_rebuild_on_remove():
    print("\n--- 移除后重建 ---")
    db = FaceDatabase()
    emb = _make_emb(0)
    db.add_face("Test", emb)
    check("添加前矩阵=None", db._embeddings_matrix is None)
    db.search(emb, threshold=0.4)
    check("首次搜索后矩阵存在", db._embeddings_matrix is not None)
    db.remove_face("Test")
    check("移除后矩阵重置", db._embeddings_matrix is None)


def run():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 50)
    print(" [GALLERY TEST] FaceDatabase Vectorized")
    print("=" * 50)
    test_vectorized_match()
    test_threshold_filter()
    test_empty_db()
    test_matrix_rebuild_on_remove()
    print(f"\n  gallery_ok={PASS} fail={FAIL}")
    return PASS, FAIL
