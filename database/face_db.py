"""
人脸数据库模块 — 管理已注册人脸的 embedding 底库。

当前使用 pickle 序列化存储，后续可无损升级为 FAISS 向量索引。
接口已预留 _faiss_index 属性，升级时只需替换内部实现。

数据结构：
  [
      {"name": "Byron", "embedding": np.ndarray(512,)},
      {"name": "Alice", "embedding": np.ndarray(512,)},
  ]

使用示例：
  db = FaceDatabase()
  db.add_face("Byron", embedding)
  db.save("face_db/identities.pkl")

  db2 = FaceDatabase()
  db2.load("face_db/identities.pkl")
  name, score = db2.search(query_embedding)
"""

import hashlib
import hmac
import os
import pickle
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np


def _compute_hmac(data: bytes, key: bytes = None) -> str:
    if key is None:
        key = hashlib.sha256(b"face_db_integrity_key").digest()
    return hmac.new(key, data, hashlib.sha256).hexdigest()


class FaceDatabase:
    """
    人脸 embedding 底库。

    职责：
      1. 存储 (name, embedding) 记录
      2. 余弦相似度检索
      3. pickle 持久化
      4. 预留 FAISS 加速接口

    线程安全说明：
      内部使用 threading.Lock 保护所有公开方法，线程安全。
    """

    def __init__(self) -> None:
        """初始化空数据库。"""
        self._records: List[Dict] = []
        self._faiss_index = None
        self._embeddings_matrix = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 增删查
    # ------------------------------------------------------------------

    def add_face(self, name: str, embedding: np.ndarray) -> None:
        """
        注册一张人脸。

        Args:
            name:      人名标识，如 "Byron"。
            embedding: 512 维归一化浮点向量 (已 L2 归一化)。

        注意：
          同名用户可多次调用以累积多个 embedding，
          检索时取最高相似度，提高姿态/光照鲁棒性。
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("embedding 必须是 numpy.ndarray")

        with self._lock:
            self._records.append({
                "name": str(name),
                "embedding": embedding.astype(np.float32),
            })
            self._embeddings_matrix = None  # invalidate
            print(f"[FaceDB] 已注册: {name} (总计 {len(self._records)} 条记录)")

    def remove_face(self, name: str) -> int:
        """
        删除指定名字的所有记录。

        Args:
            name: 要删除的人名。

        Returns:
            int: 删除的记录条数。
        """
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r["name"] != name]
            removed = before - len(self._records)
            if removed > 0:
                self._embeddings_matrix = None  # invalidate
                print(f"[FaceDB] 已删除 {name}: {removed} 条记录")
            return removed

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.4,
    ) -> Optional[Tuple[str, float]]:
        """
        在数据库中检索最匹配的人脸。

        算法：遍历所有记录，计算余弦相似度，返回最高分且超过阈值的匹配。

        Args:
            query_embedding: 查询 embedding（512 维，已归一化）。
            threshold:       余弦相似度阈值 [0.0, 1.0]。
                             低于此值的匹配不计入。

        Returns:
            (name, similarity) — 最佳匹配结果。
            None — 无匹配（数据库为空或所有记录低于阈值）。

        复杂度:
          O(N) 线性扫描。N < 10000 时性能可接受。
          N > 10000 时建议升级为 FAISS (接口已预留)。
        """
        with self._lock:
            if not self._records:
                return None

            # 向量化余弦相似度: [N,] = q @ E^T
            if self._embeddings_matrix is None or len(self._embeddings_matrix) != len(self._records):
                self._rebuild_matrix()

            if self._embeddings_matrix is not None and len(self._embeddings_matrix) > 0:
                scores = np.dot(self._embeddings_matrix, query_embedding.ravel())
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                if best_score >= threshold:
                    return (self._records[best_idx]["name"], best_score)
                return None

            # Fallback: linear scan
            best_name = None
            best_score = 0.0
            for record in self._records:
                score = float(np.dot(query_embedding, record["embedding"].ravel()))
                if score > best_score:
                    best_score = score
                    best_name = record["name"]
            if best_score >= threshold and best_name is not None:
                return (best_name, best_score)
            return None

    def _rebuild_matrix(self) -> None:
        if not self._records:
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.stack([r["embedding"].ravel() for r in self._records], axis=0)

    def get_all_names(self) -> List[str]:
        """返回所有已注册的人名（去重）。"""
        with self._lock:
            return sorted(set(r["name"] for r in self._records))

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        将数据库序列化到磁盘。

        Args:
            path: 保存路径，如 "face_db/identities.pkl"。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = [
            {"name": r["name"], "embedding": r["embedding"]}
            for r in self._records
        ]
        payload = pickle.dumps(save_data)
        hmac_sig = _compute_hmac(payload)
        with self._lock:
            with open(path, "wb") as f:
                f.write(hmac_sig.encode() + b"\n" + payload)
        print(f"[FaceDB] 已保存 {len(self._records)} 条记录 → {path}")

    def load(self, path: str) -> None:
        """
        从磁盘加载数据库。

        Args:
            path: 数据库文件路径。

        如果文件不存在，数据库保持为空（不抛异常）。
        """
        if not os.path.exists(path):
            print(f"[FaceDB] 数据库文件不存在: {path}，初始化为空")
            return

        with open(path, "rb") as f:
            content = f.read()

        if b"\n" not in content:
            raise ValueError(f"[FaceDB] 数据库文件格式无效: {path}")

        sig, payload = content.split(b"\n", 1)
        expected = _compute_hmac(payload)
        if not hmac.compare_digest(sig.decode(), expected):
            raise ValueError(f"[FaceDB] 数据库文件完整性校验失败: {path}")

        data = pickle.loads(payload)

        with self._lock:
            self._records = []
            for item in data:
                self._records.append({
                    "name": item["name"],
                    "embedding": item["embedding"].astype(np.float32),
                })
            self._embeddings_matrix = None

        print(f"[FaceDB] 已加载 {len(self._records)} 条记录 ← {path}")

    @property
    def count(self) -> int:
        """返回数据库记录总数。"""
        with self._lock:
            return len(self._records)

    @property
    def names(self) -> List[str]:
        """返回所有已注册人名的去重列表。"""
        with self._lock:
            return sorted(set(r["name"] for r in self._records))
