# AI 智能监控系统

<p align="center">
  <img src="https://img.shields.io/badge/Version-v10-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/InsightFace-0.7.3-red" alt="InsightFace">
  <img src="https://img.shields.io/badge/SCRFD-500m-cyan" alt="SCRFD">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-ONNX-orange" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/Edge-Zero_BoxMOT-brightgreen" alt="Edge">
  <img src="https://img.shields.io/badge/License-LYUN-lightgrey" alt="License">
</p>

<p align="center">
  <b>统一 AI 智能监控系统 — 人脸识别 (SCRFD + ArcFace) + 摔倒检测 (YOLOv8-Pose) 共享 Pipeline，PersonManager 唯一追踪源，边缘/桌面双部署</b>
</p>

---

## 简介

**AI 智能监控系统** 将人脸识别与摔倒检测融合为单一实时推理管线，面向边缘设备部署。

融合前两个独立项目各有自己的 Camera → Detect → Track → Render 主循环。融合后以人脸识别项目 Pipeline 为主体，摔倒检测作为 **VisionTask 插件** 嵌入，**PersonManager** 是唯一追踪数据源 —— 人脸和姿态共用 `track_id`，身份与摔倒状态可联动告警。

适用场景：
- **智慧养老** — 人脸身份识别 + 摔倒实时告警，已知身份摔倒 vs 陌生人摔倒区分
- **智能安防** — 人脸门禁 + 区域警戒 + 行为分析
- **医院监护** — 病人移动监测 + 离床告警 + 身份确认
- **边缘 AI 研究** — 多模型 Pipeline 基准测试、ONNX 后端性能对比

---

## Demo

```
┌──────────────────────────────────────────────────────────────────┐
│ FPS: 28.5    Persons: 2   Fall: 0 alert(s)    Frame: #1500       │
│ Motion: 2.3(Y) | normal(2f)  Q: 1 | Cache: 3                    │
│ cam:2ms | det:14ms | trk:1ms | rec:0ms | rnd:1ms | task.fall:42ms│
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐                             │
│   │  Byron       │    │  Unknown    │                             │
│   │  MOVING      │    │  STATIONARY │                             │
│   │  (0.87)      │    │             │                             │
│   │  ID:1        │    │  ID:5       │                             │
│   │  🦴+🟢Normal │    │  🦴+🟢Normal │                             │
│   └─────────────┘    └─────────────┘                             │
│                                                                  │
│   Press Q to quit                                                │
└──────────────────────────────────────────────────────────────────┘
```

> 人脸框 + 身份标签 (绿色=已知 / 灰色=未知) + 17 点骨架 + 摔倒状态 (红 FALL / 橙 Potential Fall / 绿 Normal)

---

## 系统架构 (融合后)

```
main.py (统一入口)
    │
    ├─ 配置加载 (统一 config.yaml, 含 face + fall 两套配置)
    │
    ├─ CameraPipeline (摄像头 → 人脸检测 → 追踪 → 识别)
    │       │
    │       ▼
    │   PersonManager (唯一追踪源: track_id, bbox, identity, history)
    │       │
    │       ├─▶ RecognitionScheduler → RecognitionWorker (ArcFace 识别)
    │       ├─▶ VisionTask: FallDetectionTask (YOLOv8-Pose 姿态 + evaluate_fall 摔倒判断)
    │       ├─▶ VisionTask: BehaviorAnalysis (可选)
    │       └─▶ VisionTask: (未来扩展...)
    │
    ├─ EventSystem (统一事件: fall_detected, stranger_alert, zone_entry...)
    │       └─ AlertManager (冷却去重: 已知身份摔倒 / 陌生人摔倒 / 摔倒恢复)
    │
    └─ Renderer (统一渲染: 人脸框 + 身份标签 + 17点骨架 + 摔倒状态)
```

**核心技术栈：**

| 阶段 | 技术 | 依赖 |
|------|------|------|
| 采集 | OpenCV VideoCapture | `opencv-python-headless` |
| 运动检测 | Frame-Difference MotionGate | `numpy` |
| 帧调度 | Adaptive FrameScheduler | (stdlib) |
| 人脸检测 | SCRFD det_500m (InsightFace) | `insightface` + `onnxruntime` |
| 姿态估计 | YOLOv8n-Pose ONNX | `onnxruntime` |
| 追踪 | LightweightIoUTracker / ByteTrack | `boxmot`(桌面) 或 `numpy`(边缘) |
| 身份管理 | PersonManager + EmbeddingCache | `numpy` |
| 人脸识别 | ArcFace w600k_mbf | `insightface` |
| 摔倒判断 | 规则方法 + 物理特征 + 滑动窗口 | `numpy` + `scipy` |
| 事件分发 | EventSystem + AlertManager | (stdlib) |
| 渲染 | OpenCV draw + imshow | `opencv-python-headless` |

---

## 功能特性

### 人脸识别 (原 ai-monitor-face-recognition v9)

- **SCRFD 人脸检测** — InsightFace det_500m ONNX, CUDA GPU 推理, 5 点关键点
- **Motion Gate** — 帧差分运动检测，跳过静止帧降低 GPU 负载
- **自适应帧调度** — 动态检测间隔 (fast=1f / normal=2f / slow=5f / force=15f)
- **多目标追踪** — ByteTrack (桌面) / LightweightIoUTracker (边缘，零 boxmot 依赖)
- **ArcFace 识别** — 512 维特征向量，余弦相似度 ≥0.70
- **异步识别 Worker** — 守护线程，非阻塞提交/收集
- **智能调度器** — 优先级: 新 track → 冷却到期 → 已识别重验证
- **人脸质量过滤** — 5 规则预筛选 (尺寸/模糊/长宽比/边界)，拒绝 ~70% 低质量人脸
- **Embedding 缓存** — LRU 128 条目，30s TTL，跨 track 恢复保留身份
- **硬去重** — 注册用户 1 框上限，重复重置 + 600 帧冷却

### 摔倒检测 (原 YOLOv8-Pose v3.5)

- **YOLOv8-Pose 姿态估计** — 17 关键点 (COCO)，输入 640×640
- **双推理后端** — Ultralytics (PyTorch/CUDA) + ONNX Runtime (CPU/边缘，零 PyTorch)
- **四路摔倒判断** — 几何规则 (AR+角度) + 物理特征 (RE/GF) + 侧倒 (AR+头部下降) + 已倒地检测
- **滑动窗口确认** — 20 帧触发比 50%，5 帧连续触发
- **持续时间确认** — 3.5s 持续才确认摔倒
- **回弹检测** — 头部回弹 15% 身高 + 2 帧 → 取消摔倒
- **恢复检测** — AR 恢复至 70% 基线自动重置
- **EMA + Savitzky-Golay** — 双平滑降噪

### 融合增强

- **PersonManager 唯一追踪** — 人脸和姿态共用 track_id，摔倒事件可关联身份
- **统一事件系统** — `fall_detected` / `fall_potential` / `fall_recovered` / `stranger_fall` / `known_person_fall`
- **摔倒+身份联动告警** — "张三摔倒!" vs "陌生人摔倒!" 区分告警
- **统一渲染** — 人脸框 + 身份标签 + 17 点骨架 + 摔倒状态 同画面叠加
- **统一配置** — 单 YAML 文件，tasks.fall_detection 独立配置节

---

## 项目结构

```
ai智能监控/
│
├── README.md                                # 本文件
├── docs/
│   ├── specs/
│   │   ├── 2026-05-16-ai-surveillance-integration-design.md   # 融合设计文档
│   │   └── 2026-05-16-ai-surveillance-integration-plan.md     # 实现计划
│   └── INTEGRATION_GUIDE.md                 # 融合使用指南
│
├── ai智能监控代码/
│   └── ai-monitor-face-recognition-main/     # ★ 融合后主项目
│       │
│       ├── main.py                           # 统一入口 (人脸识别 + 摔倒检测)
│       ├── configs/
│       │   └── default.yaml                  # 统一配置 (含 tasks.fall_detection)
│       │
│       ├── core/
│       │   ├── interfaces.py                 # VisionTask ABC + VisionEvent
│       │   ├── pipeline/pipeline.py          # 主循环 (含 _run_tasks + fall rendering)
│       │   ├── person/                       # PersonManager (唯一追踪源)
│       │   ├── detectors/                    # SCRFD 人脸检测
│       │   ├── tracking/                     # LightweightIoUTracker / ByteTrack
│       │   ├── recognition/                  # ArcFace 识别
│       │   ├── scheduler/                    # RecognitionScheduler
│       │   ├── workers/                      # 异步识别 Worker
│       │   ├── rendering/renderer.py         # 统一渲染 (人脸框 + 骨架 + 摔倒标签)
│       │   ├── event_system.py               # 事件分发
│       │   ├── alert_manager.py              # 告警管理 (含 fall 分支)
│       │   ├── face_quality.py               # 人脸质量过滤
│       │   ├── trajectory_analyzer.py        # 轨迹分析
│       │   ├── behavior_engine.py            # 行为状态机
│       │   ├── region_manager.py             # 区域管理
│       │   └── track_memory.py               # 长期轨迹记忆
│       │
│       ├── plugins/
│       │   ├── fall_detection.py             # ★ FallDetectionTask (真实实现)
│       │   └── fall_engine/                  # 摔倒检测引擎 (从 YOLOv8-Pose 项目移植)
│       │       ├── __init__.py
│       │       ├── config.py                 # 可注入配置 (替换原 config.py)
│       │       ├── fall_logic.py             # 核心摔倒判断算法
│       │       ├── features.py               # 物理特征计算 (RE/GF/AR/Angle)
│       │       ├── detection.py              # Detection / Keypoint 数据结构
│       │       └── backends/                 # 推理后端
│       │           ├── base.py               # BaseInferenceBackend ABC
│       │           ├── factory.py            # create_backend() 工厂
│       │           ├── ultralytics_backend.py # Ultralytics YOLO 后端
│       │           ├── onnx_backend.py        # ONNX Runtime 后端
│       │           └── postprocess.py         # 统一后处理
│       │
│       ├── models/
│       │   ├── scrfd_500m_bnkps.onnx         # SCRFD 人脸检测模型
│       │   └── yolov8n-pose.onnx             # YOLOv8-Pose ONNX 模型 (12.9 MB)
│       │
│       ├── database/
│       │   └── face_db.py                    # 人脸数据库 (Pickle)
│       │
│       ├── requirements/
│       │   ├── base.txt                      # numpy + pyyaml + opencv + scipy
│       │   ├── edge_cpu.txt                  # base + insightface + onnxruntime
│       │   ├── desktop.txt                   # base + insightface + onnxruntime-gpu + boxmot + ultralytics
│       │   ├── jetson.txt                    # base + insightface
│       │   └── dev.txt                       # base + pytest + black + ruff
│       │
│       ├── tests/
│       │   └── test_fall_detection_integration.py  # 摔倒检测集成测试
│       │
│       └── face_db/
│           └── identities.pkl                # 注册人脸特征
│
└── 摔倒检测/
    └── -YOLOv8-Pose--main/                   # 原摔倒检测项目 (参考)
```

---

## 快速开始

### 环境要求

| 软件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 全部 |
| VS C++ Build Tools | 2022 | Windows InsightFace 编译 |
| NVIDIA GPU | GTX 1060 6GB+ | desktop profile |
| CUDA Toolkit | 11.8+ | desktop profile |

> edge_minimal 模式不需要 GPU。

### 安装

```bash
cd ai智能监控代码/ai-monitor-face-recognition-main

# 桌面 GPU
pip install -r requirements/desktop.txt

# 边缘 CPU (树莓派/RK3588)
pip install -r requirements/edge_cpu.txt
```

### 注册人脸

```bash
python register_face.py --name YourName --simple
```

### 导出 ONNX 模型 (首次需要)

```bash
python -c "from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt').export(format='onnx', imgsz=640)"
```

### 运行

```bash
# 人脸识别 + 摔倒检测 (默认启用)
python main.py

# 仅人脸识别 (关闭摔倒检测: 修改 configs/default.yaml → tasks.fall_detection.enabled: false)
python main.py

# 边缘模式
python main.py --profile edge_minimal

# Benchmark (无渲染, 300 帧自动退出)
python main.py --benchmark

# 指定摄像头
python main.py --camera 0

# 多摄像头
python main.py --camera 0 1 2
```

### 运行测试

```bash
python tests/test_fall_detection_integration.py
```

---

## 配置

编辑 `configs/default.yaml`：

```yaml
# 人脸识别配置
detector:
  model_name: "buffalo_s"
  input_size: 640
  conf_threshold: 0.5
  detection_interval: 2
  device: cuda

recognition:
  recognition_threshold: 0.70
  recognition_cooldown: 300
  min_face_size: 48

# 摔倒检测配置
tasks:
  fall_detection:
    enabled: true                     # 开启/关闭摔倒检测
    model_path: "models/yolov8n-pose.onnx"
    device: "auto"                    # auto / cpu / cuda
    backend: "onnx"                   # onnx (边缘推荐) / ultralytics (GPU)
    interval: 5                       # 每 N 帧推理一次
    confidence_threshold: 0.5

    # 摔倒判断参数
    fall:
      horizontal_ar_threshold: 0.6
      angle_threshold: 120
      torso_inclination_threshold: 65
      min_fall_pose_duration: 3.5
      window_size: 10
      window_trigger_ratio: 0.5
      min_consecutive_triggers: 3

# 运行时开关
runtime:
  enable_event_system: true           # 事件系统 (摔倒告警需要)
  enable_alert: true                  # 告警管理
```

### 部署 Profile

| Profile | 输入 | 检测间隔 | 追踪 | GPU | 渲染 |
|---------|------|---------|------|-----|------|
| `edge_minimal` | 320px | 4f | IoU | CPU | no |
| `balanced` | 416px | 3f | IoU | CPU | yes |
| `desktop` | 640px | 2f | ByteTrack | CUDA | yes |

---

## 性能

**RTX 4060 Laptop GPU — 人脸识别 + 摔倒检测同时运行**

| 阶段 | 延迟 | 频率 |
|------|------|------|
| 人脸检测 (SCRFD 640px) | ~15ms | 每 2 帧 |
| 摔倒检测 (YOLOv8-Pose ONNX 640px) | ~42ms | 每 5 帧 |
| 人脸识别 (ArcFace) | ~5ms | 按需 (300f 冷却) |
| 追踪 + 渲染 | ~2ms | 每帧 |
| **总 (平均)** | **~15-30ms** | **~30-65 FPS** |

---

## 融合前后对比

| 维度 | 融合前 (两个独立项目) | 融合后 (统一系统) |
|------|----------------------|-------------------|
| 追踪 | 各有一套 (LightweightIoUTracker + SingleCameraTracker) | **PersonManager 唯一追踪源** |
| track_id | 各自独立分配，无法关联 | **共用 PersonManager.track_id** |
| 推理 | 人脸 ONNX CUDA，摔倒 Ultralytics PyTorch | 统一 ONNX Runtime |
| 事件 | 两套独立系统 | **EventSystem 统一分发** |
| 渲染 | 各自独立窗口 | **Renderer 统一**: 人脸框+骨架+摔倒标签 |
| 配置 | 两套 config.yaml | **统一 default.yaml** |
| 依赖 | 两套 requirements | **合并 base/desktop/edge_cpu** |
| 告警 | 无联动 | **身份+摔倒联动**: "张三摔倒!" vs "陌生人摔倒!" |

---

## VisionEvent 事件类型

| event_type | 说明 |
|---|---|
| `fall_detected` | 确认摔倒 (持续 3.5s+) |
| `fall_potential` | 疑似摔倒 (预警，待确认) |
| `fall_recovered` | 摔倒后恢复站立 |
| `stranger_fall` | 陌生人摔倒 — 特别告警 |
| `known_person_fall` | 已知人员摔倒 — 身份关联告警 |

---

## License

LYUN License

---

<p align="center">
  <sub>Built for edge AI and real-time vision monitoring.</sub>
</p>
