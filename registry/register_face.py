"""
人脸注册模块 — 支持三种模式：标准/简化/图片。
"""

import time
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from core.camera import Camera
from core.detectors import SCRFDDetector
from core.recognition import FaceRecognizer
from database import FaceDatabase


# 标准模式引导
CAPTURE_ANGLES = [
    ("正脸", "请正对摄像头，保持面部居中"),
    ("左脸", "请将脸部缓慢转向右侧，露出左脸"),
    ("右脸", "请将脸部缓慢转向左侧，露出右脸"),
    ("抬头", "请缓缓抬头，下巴微微上扬"),
    ("低头", "请缓缓低头，下巴贴近脖子"),
]
COUNTDOWN_SECONDS = 3


# ---------------------------------------------------------------------------
# 辅助：提取 embedding
# ---------------------------------------------------------------------------

def _extract_from_detections(
    recognizer: FaceRecognizer,
    frame: np.ndarray,
    faces: list,
) -> Optional[np.ndarray]:
    if not faces:
        return None
    best = max(faces, key=lambda f: f[4])
    x1, y1, x2, y2, _conf = best[0], best[1], best[2], best[3], best[4]
    crop = frame[y1:y2, x1:x2]
    return recognizer.get_embedding(crop)


# ---------------------------------------------------------------------------
# 注册统一入口：3 模式
# ---------------------------------------------------------------------------

def register(
    name: str,
    camera: Camera,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    database: FaceDatabase,
    mode: str = "standard",
    num_captures: int = 5,
    image_path: Optional[str] = None,
) -> Optional[Tuple[bool, str]]:
    if mode == "image" and image_path:
        return _register_from_image(name, image_path, recognizer, database)
    elif mode == "simple":
        return _register_simple(name, camera, detector, recognizer, database)
    else:
        return _register_standard(name, camera, detector, recognizer, database, num_captures)


# ---------------------------------------------------------------------------
# 图片注册
# ---------------------------------------------------------------------------

def _register_from_image(
    name: str,
    image_path: str,
    recognizer: FaceRecognizer,
    database: FaceDatabase,
) -> Optional[Tuple[bool, str]]:
    if not os.path.exists(image_path):
        return (False, f"文件不存在: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return (False, f"无法读取图片: {image_path}")

    embedding = recognizer.get_embedding(img)
    if embedding is None:
        return (False, "未能从图片中提取人脸特征，请确认图片包含清晰正面人脸")

    database.add_face(name, embedding)
    print(f"[Registration] ✅ 图片注册成功: {name}")
    return (True, f"注册成功: {name}")


# ---------------------------------------------------------------------------
# 简化版注册：按空格键采集，无倒计时
# ---------------------------------------------------------------------------

def _register_simple(
    name: str,
    camera: Camera,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    database: FaceDatabase,
) -> Optional[Tuple[bool, str]]:
    print(f"\n{'=' * 55}")
    print(f"  人脸注册 — 简化模式: {name}")
    print(f"  操作: 按 [SPACE] 采集  |  按 [ESC] 取消")
    print(f"  建议采集 3~5 张不同角度照片")
    print(f"{'=' * 55}\n")

    embeddings = []
    captured = 0

    with camera as cam:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            faces = detector.detect(frame)
            display = frame.copy()

            for (x1, y1, x2, y2, conf) in faces:
                color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # 提示信息
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(display, f"Name: {name} | Captured: {captured}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "[SPACE] Capture  [ENTER] Done  [ESC] Cancel",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(f"Register: {name}", display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                print("[Registration] 已取消")
                return None

            if key == 13 and captured > 0:  # ENTER
                cv2.destroyAllWindows()
                break

            if key == 32:  # SPACE
                emb = _extract_from_detections(recognizer, frame, faces)
                if emb is not None:
                    embeddings.append(emb)
                    captured += 1
                    print(f"  ✅ Captured {captured}: {name} "
                          f"(embeddings: {len(embeddings)})")
                    # 闪光效果
                    flash = np.ones_like(frame) * 255
                    cv2.imshow(f"Register: {name}", flash)
                    cv2.waitKey(100)
                else:
                    print(f"  ⚠ 未检测到人脸，请正对摄像头")

    cv2.destroyAllWindows()

    if not embeddings:
        return (False, "未采集到任何人脸，请重试")

    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    database.add_face(name, avg_embedding)

    print(f"[Registration] ✅ 注册成功: {name} ({captured} 张)")
    return (True, f"注册成功: {name}")


# ---------------------------------------------------------------------------
# 标准版注册：5 角度 + 倒计时（保留兼容）
# ---------------------------------------------------------------------------

def _register_standard(
    name: str,
    camera: Camera,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    database: FaceDatabase,
    num_captures: int = 5,
) -> Optional[Tuple[bool, str]]:
    total = min(num_captures, len(CAPTURE_ANGLES))

    print(f"\n{'=' * 55}")
    print(f"  人脸注册 — 标准模式: {name}")
    print(f"  采集: {total} 个角度，每角度 {COUNTDOWN_SECONDS}s 倒计时")
    print(f"  按 ESC 取消")
    print(f"{'=' * 55}\n")

    embeddings = []
    captured = 0

    with camera as cam:
        for i in range(total):
            angle_name, instruction = CAPTURE_ANGLES[i]
            print(f"[{i + 1}/{total}] {angle_name} — {instruction}")

            # 等待检测到人脸
            found_face = False
            while not found_face:
                ret, frame = cam.read()
                if not ret:
                    continue
                faces = detector.detect(frame)
                display = frame.copy()
                for (x1, y1, x2, y2, conf) in faces:
                    color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                _draw_guide(display, angle_name, instruction, 0, captured, total)
                cv2.imshow(f"Register: {name}", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    return None
                if faces and max(f[4] for f in faces) >= 0.5:
                    found_face = True

            # 倒计时采集
            capture_ok = False
            t0 = time.time()
            while not capture_ok:
                elapsed = time.time() - t0
                remaining = COUNTDOWN_SECONDS - int(elapsed)
                ret, frame = cam.read()
                if not ret:
                    continue
                faces = detector.detect(frame)
                display = frame.copy()
                for (x1, y1, x2, y2, conf) in faces:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                _draw_guide(display, angle_name, instruction, max(remaining, 0), captured, total)
                cv2.imshow(f"Register: {name}", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    return None
                if elapsed >= COUNTDOWN_SECONDS:
                    if not faces or max(f[4] for f in faces) < 0.5:
                        found_face = False
                        break
                    emb = _extract_from_detections(recognizer, frame, faces)
                    if emb is None:
                        t0 = time.time()
                        continue
                    embeddings.append(emb)
                    captured += 1
                    capture_ok = True
                    print(f"  ✅ {captured}/{total} — {angle_name}")
                    flash = np.ones_like(frame) * 255
                    cv2.imshow(f"Register: {name}", flash)
                    cv2.waitKey(100)
            if not capture_ok:
                i -= 1
                continue
            if captured < total:
                time.sleep(0.5)

    cv2.destroyAllWindows()
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    database.add_face(name, avg_embedding)
    print(f"[Registration] ✅ 注册成功: {name}")
    return (True, f"注册成功: {name}")


# ---------------------------------------------------------------------------
# 兼容旧接口
# ---------------------------------------------------------------------------

def capture_and_register(*args, **kwargs):
    return _register_standard(*args, **kwargs)


# ---------------------------------------------------------------------------
# 绘制辅助
# ---------------------------------------------------------------------------

def _draw_guide(frame, angle_name, instruction, countdown, captured, total):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f">>> {angle_name} <<<", (w // 2 - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, instruction, (w // 2 - 300, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if countdown > 0:
        cv2.putText(frame, str(countdown), (w // 2 - 20, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)
    cv2.putText(frame, f"Progress: {captured}/{total}",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    bar_w, bar_h = w - 40, 12
    bar_x, bar_y = 20, h - 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * captured / total), bar_y + bar_h),
                  (0, 255, 0), -1)
