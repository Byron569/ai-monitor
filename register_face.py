#!/usr/bin/env python
"""
人脸管理 CLI — 注册 / 删除 / 列出人脸。

用法:
  python register_face.py --name Byron                     # 标准注册(5角度+倒计时)
  python register_face.py --name Byron --simple            # 简化注册(按空格采集)
  python register_face.py --name Byron --image photo.jpg   # 从图片注册
  python register_face.py --remove --name Byron            # 删除
  python register_face.py --list                           # 列出所有人
"""

import argparse
import logging
import os
import sys

os.environ["ORT_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from core.camera import Camera
from core.detectors import SCRFDDetector
from core.recognition import FaceRecognizer
from database import FaceDatabase
from registry.register_face import register
from utils import get_project_root, load_config


def main():
    parser = argparse.ArgumentParser(description="Vision AI — 人脸管理工具")
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--simple", action="store_true",
                        help="简化注册: 按 SPACE 采集, 无倒计时")
    parser.add_argument("--image", type=str, default=None,
                        help="从图片注册, 如 --image photo.jpg")
    parser.add_argument("--remove", action="store_true",
                        help="删除指定人名（需配合 --name）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有已注册用户")
    parser.add_argument("--capture", "-c", type=int, default=5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    project_root = get_project_root()
    config = load_config()
    detector_cfg = config.get("detector", {})
    db_path = args.db or os.path.join(project_root, "face_db", "identities.pkl")

    db = FaceDatabase()
    db.load(db_path)

    # --- --list ---
    if args.list:
        names = db.get_all_names()
        if names:
            print(f"[System] 已注册 {len(names)} 人:")
            for n in names:
                print(f"  - {n}")
        else:
            print("[System] 数据库为空")
        return

    # --- --remove ---
    if args.remove:
        if not args.name:
            print("[Error] 需要 --name")
            sys.exit(1)
        removed = db.remove_face(args.name)
        if removed > 0:
            db.save(db_path)
            print(f"[System] 已删除 {args.name} ({removed} 条)")
        else:
            print(f"[System] 未找到: {args.name}")
        return

    # --- 注册 ---
    if not args.name:
        print("[Error] 请提供 --name, 或使用 --list / --remove / --image")
        sys.exit(1)

    db_path = os.path.join(project_root, "face_db", "identities.pkl")

    # --- 图片注册（不需要摄像头）---
    if args.image:
        recognizer = FaceRecognizer(device=args.device)
        result = register(
            name=args.name, camera=None, detector=None,
            recognizer=recognizer, database=db,
            mode="image", image_path=args.image,
        )
        if result and result[0]:
            db.save(db_path)
            print(f"[System] 已保存到 {db_path}")
        elif result:
            print(f"[System] {result[1]}")
            sys.exit(1)
        return

    # --- 摄像头注册 ---
    print(f"[System] 数据库已加载: {db.count} 条记录")

    camera = Camera(index=args.camera, width=640, height=480)
    detector = SCRFDDetector(
        model_name=detector_cfg.get("model_name", "buffalo_s"),
        input_size=detector_cfg.get("input_size", 640),
        conf_threshold=detector_cfg.get("conf_threshold", 0.3),
        device="cuda",
    )
    recognizer = FaceRecognizer(device=args.device)

    mode = "simple" if args.simple else "standard"

    try:
        result = register(
            name=args.name,
            camera=camera,
            detector=detector,
            recognizer=recognizer,
            database=db,
            mode=mode,
            num_captures=args.capture,
        )
    except KeyboardInterrupt:
        print("\n[System] 已取消")
        sys.exit(0)

    if result and result[0]:
        db.save(db_path)
        print(f"[System] 已保存到 {db_path}")
    elif result:
        print(f"[System] {result[1]}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
