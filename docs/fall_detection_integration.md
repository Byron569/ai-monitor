# Fall Detection Integration Plan

## 1. Why NOT merge YOLOv8-Pose directly into main.py

- **主循环复杂度失控**: main.py 和 pipeline.py 已经管理 10+ 个模块。直接加入 YOLOv8-Pose 推理会使主循环从 ~400 行膨胀到 ~800 行。
- **依赖爆炸**: YOLOv8-Pose 需要 `ultralytics` + `torch` + `torchvision` (~2GB)，边缘设备无法承受。
- **推理频率冲突**: 人脸检测每 2-5 帧运行，pose 估计可能需要每 5-10 帧运行。两者合并到同一调度器会互相拖累。
- **故障隔离**: Pose 模型 OOM 或推理异常不应导致人脸识别管线崩溃。
- **渐进式部署**: 可以先部署人脸识别系统，后续通过配置开关启用摔倒检测，无需重新部署整个系统。

## 2. Why VisionTask plugin

- **零依赖**: Stub 实现不需要 ultralytics/torch。只有真正启用时才需要安装。
- **独立调度**: 每个 task 有自己的 `interval` 和 `should_run()` 逻辑。
- **独立生命周期**: Task 初始化/运行/异常都隔离在 try/except 中。
- **统一事件输出**: 所有 task 通过 `VisionEvent` 输出事件，由 `EventSystem` 统一处理。
- **可替换**: 可以在不修改 pipeline 的情况下替换 FallDetectionTask 的实现（TensorRT / OpenVINO / ONNX）。

## 3. Why interval-based execution

- Pose 估计单次推理 15-30ms (CPU)/5-10ms (GPU)，每帧运行会占用全部算力。
- 跌倒是一个低频事件（持续几秒），不需要每帧检测。
- interval=5 (每 5 帧) 在 30fps 下即每秒 6 次检查，足以捕捉跌倒过程。

## 4. Why share tracks

- PersonManager 已经维护了每个人的 bbox + identity + frame_seen。
- Pose 检测应该对已跟踪的人进行，而不是重新检测画面中所有人。
- 共享 tracks 避免了"双 tracker"问题：如果 YOLOv8-Pose 自己做人检测，会与人脸 tracker 产生 ID 冲突。

## 5. Why avoid dual trackers

- 一套 tracker 维护 track_id 权威。
- 如果人脸跟踪和 Pose 跟踪各有一套 ID 系统，张三的人脸 ID=3 但 pose ID=7，无法关联。
- 解决方案：FallDetectionTask 接收 PersonManager 的 bbox 列表，在每个人的 crop 上做 pose 估计。

## 6. Future FallDetectionTask implementation plan

```python
class FallDetectionTask(VisionTask):
    def __init__(self, config):
        # 加载 YOLOv8-Pose 模型
        from ultralytics import YOLO
        self.model = YOLO(config["model"])  # yolov8n-pose.pt
        self.conf_threshold = config["confidence_threshold"]

    def run(self, frame, tracks, context):
        events = []
        for tid, bbox, ident in tracks:
            x1, y1, x2, y2 = bbox
            person_crop = frame[y1:y2, x1:x2]

            # YOLOv8-Pose 推理
            results = self.model(person_crop, verbose=False)
            keypoints = results[0].keypoints

            if keypoints is None:
                continue

            # 计算跌倒指标
            is_falling, conf = self._analyze_fall(keypoints, bbox)

            if is_falling and conf > self.conf_threshold:
                events.append(VisionEvent(
                    event_type="fall_detected",
                    track_id=tid,
                    confidence=conf,
                    payload={"bbox": bbox, "identity": ident}
                ))

        return events
```

## 7. ROI cropping strategy

- 不使用整帧输入 YOLOv8-Pose（640×480 → 多人场景下检测不准且慢）。
- 使用 PersonManager 提供的 bbox 裁剪每个人的 ROI。
- ROI 扩大 20% 缓冲区以包含可能的跌倒姿势（人体倒地后 bbox 会变宽变矮）。
- 如果某个人已经被识别，可以跳过 pose 检测（已知身份的人可以豁免）。

## 8. Reducing pose inference frequency

- **人体比例过滤**: 如果 bbox 高度 < 150px，不运行 pose（太小无法识别关键点）。
- **姿势缓存**: 如果当前 bbox 与上一帧 IoU > 0.9，复用上一帧的 pose 结果。
- **多级调度**: 
  - Level 1 (每 5 帧): 只检测 unidentified 的人
  - Level 2 (每 15 帧): 检测所有人
  - Level 3 (每 60 帧): 全画面 pose 扫描（无 bbox 的人）
- **跌倒窗口**: 只在检测到"高度变化 > 30%"时进入高频检测模式。

## 9. Fall event output

```python
VisionEvent(
    event_type="fall_detected",
    track_id=3,
    confidence=0.87,
    timestamp=1715600000.0,
    payload={
        "bbox": (100, 200, 300, 400),
        "identity": "Unknown",
        "duration_frames": 12,
        "key_angle": 45.3,  # 躯干倾斜角度
        "head_y_ratio": 0.3,  # 头部高度/人体高度比
    }
)
```

## 10. Cooldown and debounce

- 同一 track_id 的 fall_detected 事件在 30 秒内不重复触发。
- 跌倒判定需要 `min_duration_frames=10` 的连续检测窗口。
- 1 帧误检不计入（debounce: 需要连续 3 帧检测到才进入计数）。
- 事件由 `AlertManager` 统一管理冷却（当前已实现 30s cooldown）。

## 11. Dependencies (future)

```
# 当前: 零额外依赖
pip install -r requirements/edge_cpu.txt

# 未来 (启用 fall detection):
pip install ultralytics>=8.0.0
pip install torch>=2.0.0  # CPU only for edge
# 或:
pip install onnxruntime-gpu  # 如果有 GPU
```

## 12. Configuration (future)

```yaml
tasks:
  fall_detection:
    enabled: true
    interval: 5
    model: "models/yolov8n-pose.pt"
    device: cpu
    confidence_threshold: 0.6
    min_duration_frames: 10
    roi_expand_ratio: 0.2
    min_person_height: 150
    cooldown_seconds: 30
    debounce_frames: 3
```

## 13. Summary

| 决策 | 理由 |
|------|------|
| 插件化而非内联 | 故障隔离 + 按需加载 + 独立调度 |
| 空实现先占位 | 不增加当前依赖和开销 |
| 共享 tracks | 避免双 tracker ID 冲突 |
| interval 运行 | 边缘设备算力预算 |
| VisionEvent 输出 | 统一事件处理管道 |
| 配置驱动 | 无需修改代码即可升级 |
