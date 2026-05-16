# AI 智能监控系统

<p align="center">
  <img src="https://img.shields.io/badge/Version-v10-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/InsightFace-0.7.3-red" alt="InsightFace">
  <img src="https://img.shields.io/badge/SCRFD-500m-cyan" alt="SCRFD">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-ONNX-orange" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/Edge-Ready-brightgreen" alt="Edge">
  <img src="https://img.shields.io/badge/License-LYUN-lightgrey" alt="License">
</p>

<p align="center">
  <b>统一 AI 智能监控系统 — 人脸识别 (SCRFD + ArcFace) + 摔倒检测 (YOLOv8-Pose) 共享 Pipeline<br>PersonManager 唯一追踪源，边缘/桌面双部署，身份+摔倒联动告警</b>
</p>

---

## 简介

**AI 智能监控系统** 由两个项目融合而成：

| 原项目 | 核心能力 | 版本 |
|--------|---------|------|
| ai-monitor-face-recognition | SCRFD 人脸检测 + ByteTrack 追踪 + ArcFace 识别 + 行为分析 | v9.4 |
| YOLOv8-Pose Fall Detection | YOLOv8-Pose 姿态估计 + 四路摔倒判断 + 容错 Runtime | v3.5 |

融合后以人脸识别项目 Pipeline 为主体，摔倒检测作为 **VisionTask 插件** 嵌入。**PersonManager** 是唯一追踪数据源 — 人脸和姿态共用 `track_id`，身份与摔倒状态可联动告警 ("张三摔倒!" vs "陌生人摔倒!")。

适用场景：
- **智慧养老** — 人脸身份识别 + 摔倒实时告警，区分已知/陌生人摔倒
- **智能安防** — 人脸门禁 + 区域警戒 + 行为分析
- **医院监护** — 病人移动监测 + 离床告警 + 身份确认
- **零售分析** — 顾客计数 + 人口统计
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
│   │  🦴 Normal   │    │  🦴 Normal  │                             │
│   └─────────────┘    └─────────────┘                             │
│                                                                  │
│   Press Q to quit                                                │
└──────────────────────────────────────────────────────────────────┘
```

> 人脸框 + 身份标签 (绿色=已知 / 灰色=未知) + 17 点骨架 + 摔倒状态 (红 FALL / 橙 Potential Fall)

---

## 目录

1. [融合后系统架构](#融合后系统架构)
2. [融合后项目结构](#融合后项目结构)
3. [融合后技术栈](#融合后技术栈)
4. [融合前后对比](#融合前后对比)
5. [原项目一：人脸识别 (ai-monitor-face-recognition v9.4)](#原项目一人脸识别)
6. [原项目二：摔倒检测 (YOLOv8-Pose v3.5)](#原项目二摔倒检测)
7. [安装与运行](#安装与运行)
8. [配置说明](#配置说明)
9. [性能数据](#性能数据)
10. [更新日志](#更新日志)

---

## 融合后系统架构

### 单摄像头 Pipeline

```
┌────────────────────── Main Thread ─────────────────────────────────────┐
│                                                                        │
│  Camera ──▶ MotionGate ──▶ FrameScheduler ──▶ SCRFD(人脸检测)          │
│  (cv2)      (帧差分)        (自适应间隔)        InsightFace ONNX       │
│     │              │              │               │                     │
│     ▼              ▼              ▼               ▼                     │
│  [Frame]      [skip?]       [detect?]      Detection(bbox+landmarks)   │
│                                                                        │
│  SCRFD ──▶ Tracker ──▶ PersonManager(★唯一追踪源)                      │
│         (ByteTrack /     track_id + identity + embedding cache         │
│          LightIoU)             │                                       │
│             │                  ├──▶ RecognitionScheduler               │
│             │                  │       └──▶ RecognitionWorker(线程)     │
│             │                  │              └──▶ ArcFace → FaceDB    │
│             │                  │                                       │
│             │                  ├──▶ FallDetectionTask(VisionTask插件)   │
│             │                  │       └──▶ YOLOv8-Pose ONNX 全帧推理  │
│             │                  │            └──▶ 中心距离匹配 bbox      │
│             │                  │                 └──▶ evaluate_fall()  │
│             │                  │                      └──▶ VisionEvent │
│             │                  │                                       │
│             │                  ├──▶ BehaviorEngine(可选)               │
│             │                  │       └──▶ MOVING/STATIONARY/LOITERING│
│             │                  │                                       │
│             │                  └──▶ EventSystem                       │
│             │                          └──▶ AlertManager              │
│             │                                 └──▶ 身份+摔倒联动告警   │
│             │                                                          │
│             └──▶ Renderer(统一渲染)                                     │
│                     ├─── 人脸框 + 身份标签 (绿=已知 / 灰=未知)         │
│                     ├─── 17点骨架 + 关键点 (金色)                      │
│                     ├─── 摔倒状态标签 (红 FALL / 橙 Potential Fall)    │
│                     └─── 系统信息 (FPS + Persons + Fall alerts)        │
│                                                                        │
│                           │ submit(非阻塞)                              │
│                           ▼                                            │
│  ┌────────────── WORKER THREAD (daemon) ──────────────┐               │
│  │  INPUT QUEUE(max8) ──▶ InsightFace GPU ──▶ DB search│               │
│  │  单任务顺序处理          buffalo_s CUDA     O(N)线性  │               │
│  └────────────────────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────────────┘
```

**每阶段技术：**

| 阶段 | 技术 | 依赖 |
|------|------|------|
| 采集 | OpenCV VideoCapture | `opencv-python-headless` |
| 运动 | Frame-Difference MotionGate | `numpy` |
| 调度 | Adaptive FrameScheduler | (stdlib) |
| 人脸检测 | SCRFD det_500m (InsightFace) | `insightface` + `onnxruntime` |
| 姿态检测 | YOLOv8n-Pose ONNX | `onnxruntime` |
| 追踪 | LightweightIoUTracker / ByteTrack | `boxmot`(桌面) 或 `numpy`(边缘) |
| 身份 | PersonManager + EmbeddingCache | `numpy` |
| 人脸识别 | ArcFace w600k_mbf | `insightface` |
| 摔倒判断 | 四路检测 (几何+物理+侧倒+已倒地) | `numpy` + `scipy` |
| 事件 | EventSystem + AlertManager | (stdlib) |
| 渲染 | OpenCV draw + imshow | `opencv-python-headless` |

### 多摄像头 Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   MultiCameraManager                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Shared: RecognitionWorker + FaceDatabase            │   │
│  │  GlobalInferenceScheduler (max 2 concurrent detects) │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│          ┌────────────────┼────────────────┐                 │
│          ▼                ▼                ▼                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ CameraPipe   │ │ CameraPipe   │ │ CameraPipe   │        │
│  │ #cam0(thread)│ │ #cam1(thread)│ │ #cam2(thread)│        │
│  │              │ │              │ │              │        │
│  │ OWN: tracker │ │ OWN: tracker │ │ OWN: tracker │        │
│  │      memory  │ │      memory  │ │      memory  │        │
│  │      person  │ │      person  │ │      person  │        │
│  │      metrics │ │      metrics │ │      metrics │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 融合后项目结构

```
ai智能监控/
│
├── README.md                                    # 本文件
├── docs/
│   ├── specs/
│   │   ├── 2026-05-16-ai-surveillance-integration-design.md
│   │   └── 2026-05-16-ai-surveillance-integration-plan.md
│   └── INTEGRATION_GUIDE.md
│
├── ai智能监控代码/
│   └── ai-monitor-face-recognition-main/         # ★ 融合后主项目
│       │
│       ├── main.py                               # 统一入口
│       ├── register_face.py                      # 人脸注册 CLI
│       │
│       ├── configs/
│       │   ├── default.yaml                      # 统一配置
│       │   ├── edge_minimal.yaml                 # 320px/CPU/IoU
│       │   ├── balanced.yaml                     # 416px/CPU/IoU
│       │   └── desktop.yaml                      # 640px/CUDA/ByteTrack
│       │
│       ├── core/
│       │   ├── interfaces.py                     # VisionTask ABC + VisionEvent
│       │   ├── face_quality.py                   # 人脸质量过滤 (5规则)
│       │   ├── camera/                           # 摄像头采集
│       │   ├── detectors/                        # SCRFD 人脸检测
│       │   ├── recognition/                      # ArcFace 识别
│       │   ├── tracking/                         # LightweightIoUTracker / ByteTrack
│       │   ├── person/                           # PersonManager + EmbeddingCache
│       │   ├── pipeline/pipeline.py              # 主循环 (_run_tasks + fall rendering)
│       │   ├── rendering/renderer.py             # 统一渲染 (人脸框+骨架+摔倒标签)
│       │   ├── scheduler/                        # RecognitionScheduler
│       │   ├── workers/                          # 异步 RecognitionWorker
│       │   ├── track_memory.py                   # 长期轨迹记忆 (Hungarian)
│       │   ├── track_reassociation.py            # 丢失 track 恢复 (IoU→spatial→none)
│       │   ├── frame_scheduler.py                # 自适应检测间隔调度
│       │   ├── trajectory_analyzer.py            # 轨迹分析 (速度/方向)
│       │   ├── behavior_engine.py                # 行为状态机
│       │   ├── region_manager.py                 # 区域警戒
│       │   ├── event_system.py                   # 事件分发
│       │   ├── alert_manager.py                  # 告警管理 (含 fall 分支)
│       │   └── metrics_logger.py                 # 统一指标
│       │
│       ├── plugins/
│       │   ├── fall_detection.py                 # ★ FallDetectionTask (真实实现 ~220行)
│       │   └── fall_engine/                      # 摔倒检测引擎
│       │       ├── __init__.py
│       │       ├── config.py                     # 可注入配置 (替换原全局 config.py)
│       │       ├── fall_logic.py                 # 核心摔倒判断 (469行)
│       │       ├── features.py                   # 物理特征 (RE/GF/AR/Angle/HD)
│       │       ├── detection.py                  # Detection/Keypoint 数据结构
│       │       └── backends/                     # 推理后端
│       │           ├── base.py                   # BaseInferenceBackend ABC
│       │           ├── factory.py                # create_backend() 工厂
│       │           ├── ultralytics_backend.py     # Ultralytics YOLO 后端
│       │           ├── onnx_backend.py            # ONNX Runtime 后端
│       │           └── postprocess.py             # 统一后处理
│       │
│       ├── models/
│       │   ├── scrfd_500m_bnkps.onnx             # SCRFD 人脸检测模型
│       │   └── yolov8n-pose.onnx                 # YOLOv8-Pose ONNX (12.9 MB)
│       │
│       ├── database/
│       │   └── face_db.py                        # 人脸数据库 (Pickle, 向量化搜索)
│       │
│       ├── requirements/
│       │   ├── base.txt                          # numpy + pyyaml + opencv + scipy
│       │   ├── edge_cpu.txt                      # base + insightface + onnxruntime
│       │   ├── desktop.txt                       # base + insightface + onnx-gpu + boxmot + ultralytics
│       │   ├── jetson.txt                        # base + insightface
│       │   └── dev.txt                           # base + pytest + black + ruff
│       │
│       ├── tests/
│       │   ├── run_all_tests.py                  # 全量测试 (188 checks / 18 modules)
│       │   ├── test_fall_detection_integration.py # 摔倒检测集成测试
│       │   └── core/ (17 test modules)
│       │
│       └── face_db/
│           └── identities.pkl                    # 注册人脸特征
│
└── 摔倒检测/
    └── -YOLOv8-Pose--main/                       # 原摔倒检测项目 (参考)
```

---

## 融合后技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 人脸检测 | SCRFD det_500m (InsightFace) | 0.7.3 | 轻量人脸检测 + 5点关键点 |
| 人脸识别 | ArcFace ONNX (InsightFace) | 0.7.3 | 512维特征提取 |
| 识别模型 | buffalo_s (w600k_mbf) | — | 移动端轻量骨干网络 |
| 姿态估计 | YOLOv8n-Pose ONNX | 8.x | 17关键点人体姿态 |
| 追踪 (桌面) | ByteTrack (BoxMOT) | 18.x | Kalman + 级联匹配 |
| 追踪 (边缘) | LightweightIoUTracker | — | 贪心 IoU，零依赖 |
| 轨迹记忆 | TrackMemory (Hungarian) | — | 全局最优分配 + 锁 |
| 人脸质量 | FaceQualityFilter | — | 5规则预筛选 (尺寸/模糊/长宽比) |
| Embedding 缓存 | LRU OrderedDict | — | 128条目, 30s TTL |
| 运动检测 | MotionGate (帧差分) | — | 像素级运动门控 |
| 摔倒判断 | 四路检测 + 滑动窗口 | — | 几何+物理+侧倒+已倒地 |
| 物理特征 | EMA + Savitzky-Golay | — | RE / GF / HD 双平滑 |
| 推理引擎 | ONNX Runtime | ≥ 1.15 | GPU/CPU 模型推理 |
| 图像处理 | OpenCV | ≥ 4.8 | 采集、显示、绘制 |
| 数值计算 | NumPy + SciPy | ≥ 1.24 / ≥ 1.10 | 矩阵运算 + 信号平滑 |
| 配置 | YAML | ≥ 6.0 | 多 Profile 级联合并 |
| 语言 | Python | 3.12 | 应用逻辑 |

---

## 融合前后对比

| 维度 | 融合前 (两个独立项目) | 融合后 (统一系统) |
|------|----------------------|-------------------|
| **架构** | 各自独立 main.py + 主循环 | 人脸项目 Pipeline 为主体，摔倒作为 VisionTask 插件 |
| **追踪** | 各有一套 (LightweightIoUTracker + SingleCameraTracker) | **PersonManager 唯一追踪源**，摔倒不再自带追踪 |
| **track_id** | 各自独立分配，无法关联 | **共用 PersonManager.track_id**，Face ID ↔ Pose 联动 |
| **推理** | 人脸 ONNX CUDA，摔倒 Ultralytics PyTorch | 统一 ONNX Runtime，一套模型管理 |
| **后端** | 人脸仅 ONNX，摔倒 Ultralytics+ONNX | **create_backend() 工厂**，onnx/ultralytics 一键切换 |
| **事件** | 两套独立系统 | **EventSystem 统一分发** fall_detected / stranger_alert |
| **告警** | 无联动 | **身份+摔倒联动**: "张三摔倒!" vs "陌生人摔倒!" |
| **渲染** | 各自独立窗口 | **Renderer 统一**: 人脸框+骨架+摔倒标签同画面 |
| **配置** | 两套 config.yaml | **统一 default.yaml**，tasks.fall_detection 独立节 |
| **依赖** | 两套 requirements | **合并 base/desktop/edge_cpu/jetson**，scipy 进 base |
| **摄像头** | 各自单/双摄像头模式 | 统一多摄像头管理 (GlobalInferenceScheduler) |
| **部署** | 各自 profile 体系 | 统一 edge_minimal / balanced / desktop 三 profile |

---

## 原项目一：人脸识别

### 概述

**AI Smart Monitoring System v9.4** — 生产级实时边缘 AI 人脸识别管线。SCRFD 检测 + ByteTrack 追踪 + ArcFace 识别 + 行为分析层，全部本地硬件实时运行。

### 工作流程

```
Camera → MotionGate(运动检测跳过静止帧)
       → FrameScheduler(自适应检测间隔: 1f/2f/5f/15f)
       → SCRFD(人脸检测, bbox + 5点关键点, ONNX CUDA)
       → LightweightIoUTracker / ByteTrack(多目标追踪)
       → TrackMemory(Hungarian 全局匹配 + 锁机制)
       → PersonManager(唯一身份源: track_id → identity + embedding缓存)
       → FaceQualityFilter(5规则: 尺寸/模糊/长宽比/边界/综合)
       → RecognitionScheduler(优先级: 新track → 冷却到期 → 已识别重验证)
       → RecognitionWorker(异步线程, ArcFace 512d 特征提取)
       → FaceDB(向量化 np.dot 单次余弦相似度搜索)
       → BehaviorEngine(可选, MOVING/STATIONARY/LOITERING/DISAPPEARED)
       → RegionManager(可选, 区域 entry/leave 事件)
       → EventSystem + AlertManager(可选, 30s 冷却去重)
       → Renderer(人脸框 + 身份标签 + 行为状态 + FPS)
```

### 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 人脸检测 | SCRFD det_500m (InsightFace) | 0.7.3 | 轻量人脸检测 + 5点关键点 |
| 追踪 (桌面) | ByteTrack (BoxMOT) | 18.x | Kalman + 级联匹配 |
| 追踪 (边缘) | LightweightIoUTracker | — | 贪心 IoU，零 boxmot |
| 轨迹记忆 | TrackMemory (Hungarian) | — | 全局最优分配 + lock/unlock |
| 轨迹恢复 | TrackReassociation | — | IoU → spatial → none 三级 |
| 人脸识别 | ArcFace ONNX (InsightFace) | 0.7.3 | 512维特征提取 |
| 识别模型 | buffalo_s (w600k_mbf) | — | 移动端轻量骨干 |
| 人脸质量 | FaceQualityFilter | — | 5规则预筛选 (~70%拒绝率) |
| Embedding Cache | LRU OrderedDict | — | 128条目, 30s TTL |
| 运动检测 | MotionGate (帧差分) | — | threshold=2.0 |
| 帧调度 | Adaptive FrameScheduler | — | fast/normal/slow/force |
| 行为分析 | TrajectoryAnalyzer + BehaviorEngine | — | MOVING/STATIONARY/LOITERING |
| 区域警戒 | RegionManager (point-in-polygon) | — | Zone entry/leave |
| 事件系统 | EventSystem + AlertManager | — | 统一事件 + 30s cooldown |
| 触发器 | overlap / out-of-frame / separation | — | 身份重置 + 强制重识别 |
| 推理引擎 | ONNX Runtime GPU/CPU | ≥ 1.15 | CUDAExecutionProvider |
| 图像处理 | OpenCV | ≥ 4.8 | 采集、显示、渲染 |
| 数值 | NumPy | ≥ 1.24 | 矩阵运算 |
| 配置 | YAML | ≥ 6.0 | 3 Profile 级联合并 |
| 语言 | Python | 3.12 | 应用逻辑 |

### 核心设计决策

| 决策 | 原因 |
|------|------|
| 异步识别 Worker 线程 | ArcFace 推理 (5-8ms) 永不阻塞视频渲染 |
| 运动门控检测 | 静止帧跳过 SCRFD，显著降低 GPU 负载 |
| 自适应检测间隔 | fast=1f (运动) / normal=2f / slow=5f (静止) / force=15f (最大跳过) |
| Hungarian 全局匹配 + 锁 | 锁定已确认匹配，防止交叉时 ID 互换 |
| 无速度预测 | 由于检测间隔可变，改为依赖中心距离+尺寸权重 |
| 硬去重 | 注册用户 1 框上限，重复重置 + 600f 冷却 |
| Embedding 缓存 (10s TTL) | 丢失 track 的 embedding 缓存，新 track 自动匹配恢复身份 |
| 分离触发器 (0.5s) | 重叠分离后解锁 Hungarian 锁 + 强制重识别 |
| 单任务 Worker 队列 | 防止多人脸 GPU 显存竞争 |
| 300帧识别冷却 | 相比每帧识别减少 10-30x ArcFace 调用 |
| 线程安全设计 | 主线程拥有 PersonManager；Worker 线程只读 FaceDB |

### 功能特性

**核心管线：**
- SCRFD 人脸检测 — InsightFace det_500m ONNX, CUDA GPU 推理, 5点关键点
- Motion Gate — 帧差分运动检测 (threshold 2.0)，跳过静止帧
- 自适应帧调度 — 动态检测间隔 (fast=1f / normal=2f / slow=5f / force=15f)
- ByteTrack 追踪 — Kalman 滤波 + Hungarian 全局匹配，稳定 ID (桌面)
- Lightweight IoU Tracker — 纯 Python + numpy，零 boxmot (边缘)
- TrackMemory — 长期轨迹记忆，方向惩罚、锁机制、尺寸权重
- Track Reassociation — 三级恢复 (IoU → spatial → none)
- ArcFace 识别 — 512维特征 (buffalo_s)
- 异步识别 Worker — 守护线程，非阻塞提交/收集
- 智能调度器 — 优先级: 新 track → 冷却到期 → 已识别重验证
- Embedding 缓存 — LRU 128条目，30s TTL
- 硬去重 — 注册用户 1 框上限

**行为分析层：**
- TrajectoryAnalyzer — per-track 速度/方向/静止帧累计
- BehaviorEngine — 状态机: MOVING / STATIONARY / LOITERING / DISAPPEARED
- RegionManager — 区域系统，point-in-polygon entry/leave 追踪
- EventSystem — 统一事件分发
- AlertManager — 30s 冷却告警去重

**触发器系统：**
- Out-of-frame trigger — track 离开画面边界强制重识别
- Overlap trigger — IoU > 0.3 时立即重置身份
- Separation trigger — 分离 0.5s 后解锁 Hungarian + 强制重识别
- Hard dedup trigger — 重复注册名重置 + 600帧冷却

**识别优化 (v9)：**
- FaceQualityFilter — 5规则预筛选 (尺寸/模糊/长宽比/边界)，拒绝 ~70% 低质量
- Identity Cooldown — 已识别 600f 冷却 (vs 未知 300f)
- Failed Backoff — per-track 指数退避 (90f × fail_count)，上限 20 次
- Queue Pressure Gate — Worker 队列 ≥3 时仅允许新 track
- Embedding Cache — LRU 128, 30s TTL
- 向量化搜索 — numpy `np.dot(matrix, query)` O(N×512) 单次余弦

**测试与工程：**
- 209 测试全通过 (20 模块)，零相机/GPU/模型下载依赖
- 3 部署 Profile: edge_minimal(320px/CPU/IoU) / balanced(416px/CPU) / desktop(640px/CUDA/ByteTrack)
- 分层依赖: requirements/{base,edge_cpu,desktop,jetson,dev}.txt
- 每阶段延迟计时: SCRFD pre/infer/post
- GPU/CPU 双模式: 自动检测 CUDAExecutionProvider
- YAML 配置: 全部可调参数 + profile 级联 (default → profile → CLI args)
- CLI 人脸管理: register/list/remove
- 线程安全架构

### 部署 Profile

| Profile | 输入 | 检测间隔 | 追踪 | GPU | 渲染 |
|---------|------|---------|------|-----|------|
| `edge_minimal` | 320px | 4f | IoU | CPU (onnx) | no |
| `balanced` | 416px | 3f | IoU | CPU (onnx) | yes |
| `desktop` | 640px | 2f | ByteTrack | CUDA (onnx-gpu) | yes |

### 性能 (RTX 4060 Laptop)

| 阶段 | 延迟 | 频率 |
|------|------|------|
| Camera | 2ms | 每帧 |
| Motion Gate | <1ms | 每帧 |
| SCRFD Detection (640px) | ~15ms | 自适应 (1-15f) |
| ByteTrack + Hungarian | ~1ms | 每检测帧 |
| Recognition (buffalo_s) | 5ms | 按需 (300f 冷却) |
| Behavior Analysis | <1ms | 每帧 |
| Render | 1ms | 每帧 |
| **Total (avg)** | **~6-10ms** | **~100-160 FPS*** |

*\*实际 FPS 受摄像头帧率限制 (通常 30 FPS)。Motion gate 跳过 ~70% 帧*

---

## 原项目二：摔倒检测

### 概述

**AI Fall Detection System v3.5** — 生产级容错边缘 AI 摔倒检测系统。YOLOv8-Pose 姿态估计 + ByteTrack 追踪 + 四路摔倒判断 + ONNX Runtime 后端 + 完整容错 Runtime Engine。

### 工作流程

```
Camera ──▶ Backend.infer() ──▶ Detection(dataclass)
              (YOLO/ONNX)            │
                                     ▼
                          Tracking.update()
                          (IoU matching + ghost)
                                     │
                                     ▼
                          TrackState(dataclass)
                                     │
                                     ▼
                          evaluate_fall()
                          (四路检测逻辑)
                              ├─ 几何规则: AR + 角度
                              ├─ 物理特征: RE/GF
                              ├─ 侧倒检测: AR + Head Descent
                              └─ 已倒地检测: 持续水平姿态
                                     │
                                     ▼
                          EventRuntime(cooldown + dedup)
                                     │
                                     ▼
                          Event(dataclass) ──▶ Serializers ──▶ JSON
```

**摔倒判断详细流程：**

```
17关键点 → yolo_to_5keypoints() (H/N/B/KL/KR)
       → compute_all_features()
            ├─ ratio_bbox (宽高比)
            ├─ log_angle (身体垂直夹角)
            ├─ re (旋转能量, 倒立摆模型)
            ├─ gf (重力因子, 重心加速度)
            ├─ ratio_derivative (宽高比变化率)
            └─ head_descent (头部下降)
       → evaluate_fall() 四路检测:
            ├─ Path 1 — 规则判断: AR < horizontal_ar_threshold AND Angle < angle_threshold
            ├─ Path 2 — 物理判断: RE > re_threshold OR GF > gf_threshold
            ├─ Path 3 — 侧倒判断: AR + HeadDescent > head_descent_threshold
            └─ Path 4 — 已倒地: 快速通道 (高置信度跳过持续时间)
       → 滑动窗口确认: 20帧窗口, 50%触发比, 5帧连续触发
       → 持续时间确认: 3.5s持续 → confirmed FALL
       → 回弹检测: 头部回弹15%身高+2帧 → 取消fall
       → 恢复检测: AR恢复至70%基线 → auto-reset
```

### 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 姿态估计 | YOLOv8-Pose (nano) | 8.x | 17关键点人体姿态 |
| 追踪 | ByteTrack (Ultralytics) | 8.x | Kalman + 级联匹配 + Ghost机制 |
| 摔倒判断 | 自定义四路逻辑 | — | 几何+物理+侧倒+已倒地 |
| 物理特征 | EMA + Savitzky-Golay | — | RE / GF / HD 双平滑降噪 |
| 跨摄像头 | HSV Histogram + Hungarian | — | 上半身色彩直方图匹配 |
| ONNX 推理 | ONNX Runtime | ≥ 1.14 | CPU/CUDA 后端 |
| 图像处理 | OpenCV | ≥ 4.8 | 采集、显示、绘制 |
| 配置 | YAML | ≥ 6.0 | 全部参数 config.yaml |
| 语言 | Python | 3.12 | 应用逻辑 |

### 核心设计决策

| 决策 | 原因 |
|------|------|
| 纯对象 Pipeline (零 dict) | 类型安全，阶段入口验证，未来 async-safe |
| 后端抽象层 | 一键切换 ultralytics / ONNX / 未来后端 |
| EventBus 广播 (非直接返回) | 未来 WebSocket / MQTT / Kafka 可直接订阅 |
| StateMachine 拒绝非法转换 | Runtime 自愈，混沌测试抗性 |
| 调度器驱动 (非直接调用) | 不同 Pipeline 不同 FPS，多模型就绪 |
| Worker 隔离 + 异常边界 | 单个 Worker 崩溃不影响整个 Runtime |
| SharedFrameBuffer + ref counting | 零拷贝设计，未来多模型帧共享 |
| 单序列化出口 | 零 dict 泄漏，JSON 友好保证 |
| 阶段入口 Validator | 立即拒绝无效对象，无静默转换 |
| Session 导向 (非全局状态) | 多摄像头隔离，per-session 生命周期 |

### 功能特性

**核心检测管线：**
- YOLOv8-Pose — 17关键点人体姿态 (Ultralytics)
- ByteTrack 追踪 — Kalman + 级联匹配，稳定 ID
- Ghost Target 系统 — 丢失 track 保留 + fall-state 继承
- ROI 二次推理 — 低置信度潜在摔倒重检测
- 自适应帧调度 — `inference_interval` 跳帧 (低功耗)
- Multi-Person Limiting — `max_persons` 上限

**四路摔倒检测：**
- Path 1 — 几何规则: AR < 阈值 AND 角度 < 阈值
- Path 2 — 物理特征: RE > 阈值 OR GF > 阈值
- Path 3 — 侧倒: AR + HeadDescent > 阈值
- Path 4 — 已倒地: 快速通道 (高置信度跳过持续时间)
- 滑动窗口 — 20帧触发比 50%，5帧连续触发
- 持续时间确认 — 3.5s 持续 → confirmed FALL
- 回弹检测 — 头部回弹 15% 身高 + 2帧 → 取消
- 状态粘性 — 已确认 fall 在信号丢失时保持
- 恢复检测 — AR 恢复至 70% 基线自动重置

**物理特征：**
- Rotational Energy (RE) — 倒立摆模型，帧率归一化 (rad/s)
- Gravity Factor (GF) — 重心向地加速度 (pixel/s²)
- Head Descent (HD) — 长期头部相对身高下降
- EMA + Savitzky-Golay — 双平滑降噪
- Torso Inclination — 髋→肩向量角度

**双摄像头：**
- 多进程架构 — Queue 桥接独立摄像头进程
- HSV 直方图匹配 — 上半身跨摄像头匹配
- Stable Marriage 算法 — Gale-Shapley 全局最优
- 双重确认 — 双摄像头确认 → 95%置信度，单摄像头 → 60%

**推理后端：**
- Ultralytics Backend — 原生 YOLOv8-Pose + ByteTrack + PyTorch
- ONNX Runtime Backend — 零 PyTorch，CPU/CUDA
- 后端工厂 — `create_backend()` 一键切换
- 预留: OpenVINO / NCNN / TensorRT

**容错 Runtime Engine (v3.4-v3.5)：**
- RuntimeStateMachine — 11 状态 (INITIALIZING→WARMING_UP→RUNNING→DEGRADED→BACKPRESSURE→OVERLOADED→RECOVERING→RESTARTING→STOPPING→STOPPED→FAILED)
- RuntimePolicy — overload / reconnect / frame drop / cooldown / retry / restart / degradation
- BackpressureController — 自动帧丢弃 / 队列裁剪 / FPS 降级
- DegradationController — 5 级自动降级 (normal→reduce_fps→reduce_res→disable_vis→minimal)
- FaultRecovery — backend restart / worker restart / camera reconnect / session recovery
- ResourceManager — CPU / RAM / queue / thermal 实时监控
- RuntimeScheduler + RuntimeWorker + RuntimeClock + RuntimeDiagnostics
- EventBus + RuntimeSignals (EventSignal / HealthSignal / ErrorSignal / LifecycleSignal / CameraSignal)
- SharedFrameBuffer (ring buffer + ref counting)

**纯对象 Runtime Core (v3.2-v3.3)：**
- Detection / Keypoint / TrackState / Event / FrameContext dataclass
- DetectionPipeline — 6 阶段编排器 (infer→Detect→Track→Fall→Event→Serialize)
- EventRuntime — 事件生成 + cooldown 去重
- Serializers — 唯一 object→dict 出口
- Validators — 阶段入口 bbox/score/event_type 验证

**测试与工具：**
- 227 测试全通过 (10 模块)，零相机/GPU/模型下载依赖
- `export_onnx.py` — PT → ONNX 导出 (opset, dynamic, simplify)
- `benchmark_backend.py` — 多后端 FPS/latency/memory 对比
- `check_runtime_purity.py` — 自动扫描 dict/numpy 泄漏
- `runtime_chaos_test.py` — 混沌注入 (camera disconnect / CPU spike / queue overflow / backend crash)

### 性能

**后端对比：**

| 指标 | Ultralytics | ONNX |
|------|-------------|------|
| FPS | 12.4 | 24.7 |
| 平均延迟 | 80.6ms | 40.5ms |
| 内存 | 1200MB | 420MB |

> ONNX vs Ultralytics: FPS 2.0x, 内存节省 780MB

### SDK 使用

```python
from fall_detection import FallDetector
import cv2

detector = FallDetector(
    model_path="yolov8n-pose.pt",
    device="cpu",
    input_size=320,
    inference_interval=2,
    enable_visualization=False,
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    result = detector.process_frame(frame, camera_id="cam_0")
    for event in result["events"]:
        if event["event_type"] == "fall_confirmed":
            print(f"Fall: track={event['track_id']} conf={event['confidence']:.2f}")
```

---

## 安装与运行

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

# 桌面 GPU (推荐)
pip install -r requirements/desktop.txt

# 边缘 CPU (树莓派 / RK3588)
pip install -r requirements/edge_cpu.txt

# 编译 InsightFace (Windows)
pip install insightface-0.7.3\insightface-0.7.3
```

### 导出 ONNX 模型 (首次)

```bash
python -c "from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt').export(format='onnx', imgsz=640)"
```

### 注册人脸

```bash
python register_face.py --name YourName --simple     # 摄像头拍照注册
python register_face.py --name Alice --image a.jpg   # 从照片注册
python register_face.py --list                        # 列出所有人脸
python register_face.py --remove --name Alice         # 删除
```

### 运行

```bash
# 人脸识别 + 摔倒检测 (默认启用)
python main.py

# 仅人脸识别 (关闭摔倒)
# 编辑 configs/default.yaml → tasks.fall_detection.enabled: false

# GPU 桌面全功能
python main.py --profile desktop

# 边缘最低功耗 (无显示)
python main.py --profile edge_minimal

# CPU 平衡模式
python main.py --profile balanced

# 指定摄像头
python main.py --camera 0

# Benchmark (自动无渲染, 300帧)
python main.py --benchmark

# 多摄像头
python main.py --camera 0 1 2
```

### 运行测试

```bash
# 全量测试
python tests/run_all_tests.py

# 核心测试
python tests/run_core_tests.py

# 摔倒检测集成测试
python tests/test_fall_detection_integration.py
```

### 按键控制

| 键 | 功能 |
|----|------|
| `q` | 退出 |
| `ESC` | 取消注册 |

---

## 配置说明

### 统一配置文件

编辑 `configs/default.yaml`：

```yaml
# 摄像头
camera:
  index: 0
  width: 640
  height: 480

# 人脸检测
detector:
  model_name: "buffalo_s"
  input_size: 640
  conf_threshold: 0.5
  nms_threshold: 0.4
  detection_interval: 2
  device: cuda

# 人脸识别
recognition:
  model_name: "buffalo_s"
  recognition_threshold: 0.70
  recognition_cooldown: 300
  recognized_cooldown: 600
  failed_backoff: 90
  max_attempts: 20
  min_face_size: 48
  blur_threshold: 80
  min_quality_score: 0.55
  max_queue_size: 4
  queue_pressure_threshold: 3
  embedding_cache_ttl: 30
  max_cache_size: 128

# 追踪
tracking:
  type: iou                               # iou (边缘) / bytetrack (桌面)
  iou_threshold: 0.3
  max_lost: 15
  min_hits: 2

# 摔倒检测 (VisionTask 插件)
tasks:
  fall_detection:
    enabled: true
    model_path: "models/yolov8n-pose.onnx"
    device: "auto"                        # auto / cpu / cuda
    backend: "onnx"                       # onnx (边缘) / ultralytics (GPU)
    interval: 5                           # 每 N 帧推理一次
    input_size: 640
    confidence_threshold: 0.5
    roi_min_height: 150

    # 摔倒判断参数
    fall:
      horizontal_ar_threshold: 0.6
      angle_threshold: 120
      torso_inclination_threshold: 65
      min_fall_pose_duration: 3.5
      re_threshold: 20
      gf_threshold: 8000
      head_descent_threshold: 0.2
      window_size: 10
      window_trigger_ratio: 0.5
      min_consecutive_triggers: 3
      min_standing_ar: 1.2
      min_bbox_area: 3000
      ema_alpha: 0.3
      sg_window: 7
      sg_polyorder: 2
    features:
      head_descent_window: 20
      head_descent_min_pixels: 15
    tracking:
      ghost_timeout: 3.0
      fallen_ghost_timeout: 5.0
      distance_threshold: 150
      history_length: 36
    cross_camera:
      hist_threshold: 0.5
    camera_process:
      roi_conf: 0.25
      roi_enabled: true

# 运动检测
motion:
  threshold: 2.0
  history: 0
  force_interval: 15

# 行为分析
behavior:
  stationary_threshold: 60
  loitering_threshold: 300
  alert_cooldown: 30

# 运行时开关
runtime:
  mode: edge_minimal
  use_gpu: true
  enable_behavior: false
  enable_event_system: true               # 摔倒告警需要
  enable_alert: true                      # 告警管理
  enable_reassociation: false
  enable_region: false
```

### Profile 配置

| Profile | 输入 | 检测间隔 | 追踪 | GPU | 渲染 |
|---------|------|---------|------|-----|------|
| `edge_minimal` | 320px | 4f | IoU | CPU (onnx) | no |
| `balanced` | 416px | 3f | IoU | CPU (onnx) | yes |
| `desktop` | 640px | 2f | ByteTrack | CUDA (onnx-gpu) | yes |

**优先级: CLI args > profile config > default.yaml**

### 快速调参

| 目标 | 设置 |
|------|------|
| 最大 FPS | `motion.enabled=true, detector.detection_interval=5` |
| 最佳精度 | `conf_threshold=0.3, recognition_threshold=0.8` |
| 低显存 GPU | `motion.enabled=true, detection_interval=2` |
| CPU Only | `--device cpu, motion.enabled=true, detect slow` |
| 减少误报 | 增大 `min_fall_pose_duration` 到 5.0 |
| 更快检测摔倒 | 减小 `min_fall_pose_duration`, 增大 `window_trigger_ratio` |
| 更多告警 | 减小 `alert_cooldown`, 增加 zones |

---

## 性能数据

### 融合后 (RTX 4060 Laptop, 人脸+摔倒同时运行)

| 阶段 | 延迟 | 频率 |
|------|------|------|
| 人脸检测 (SCRFD 640px) | ~15ms | 每 2 帧 |
| 摔倒检测 (YOLOv8-Pose ONNX 640px) | ~42ms | 每 5 帧 |
| 人脸识别 (ArcFace) | ~5ms | 按需 (300f 冷却) |
| 追踪 + 渲染 | ~2ms | 每帧 |
| **总 (平均)** | **~15-30ms** | **~30-65 FPS** |

### 融合前 — 人脸识别 (RTX 4060 Laptop, v8)

| 阶段 | 延迟 | 频率 |
|------|------|------|
| Camera | 2ms | 每帧 |
| Motion Gate | <1ms | 每帧 |
| SCRFD Detection (640px) | ~15ms | 自适应 (1-15f) |
| ByteTrack + Hungarian | ~1ms | 每检测帧 |
| Recognition (buffalo_s) | 5ms | 按需 (300f 冷却) |
| Behavior Analysis | <1ms | 每帧 |
| Render | 1ms | 每帧 |
| **Total (avg)** | **~6-10ms** | **~100-160 FPS** |

### 融合前 — 摔倒检测 (后端对比)

| 指标 | Ultralytics | ONNX |
|------|-------------|------|
| FPS | 12.4 | 24.7 |
| 平均延迟 | 80.6ms | 40.5ms |
| 内存 | 1200MB | 420MB |

### 检测频率影响

| 模式 | 间隔 | GPU 负载 | 平均延迟 |
|------|------|----------|----------|
| fast | 每帧 | 100% | ~20ms |
| normal | 每 2 帧 | 50% | ~12ms |
| slow | 每 5 帧 | 20% | ~8ms |
| motion-gated | 自适应 | ~30% | ~6-10ms |

---

## VisionEvent 事件类型

| event_type | 说明 |
|---|---|
| `fall_detected` | 确认摔倒 (持续 3.5s+) |
| `fall_potential` | 疑似摔倒 (预警，待确认) |
| `fall_recovered` | 摔倒后恢复站立 |
| `stranger_fall` | 陌生人摔倒 — 特别告警 |
| `known_person_fall` | 已知人员摔倒 — "张三摔倒!" |

---

## 更新日志

### v10 — 项目融合 (2026-05-16)

- **融合** 人脸识别 + 摔倒检测为统一系统
- **新增** `plugins/fall_engine/` — 摔倒检测引擎 (fall_logic.py, features.py, backends/, detection.py)
- **新增** `plugins/fall_engine/config.py` — 可注入配置模块 (替换原全局 config.py)
- **新增** `plugins/fall_detection.py` — FallDetectionTask 真实实现 (~220行)
- **新增** `models/yolov8n-pose.onnx` — 导出的 ONNX 模型 (12.9 MB)
- **新增** `core/rendering/renderer.py` — 骨架绘制 + 摔倒状态标签
- **新增** `tests/test_fall_detection_integration.py` — 摔倒检测集成测试
- **新增** `docs/INTEGRATION_GUIDE.md` — 融合使用指南
- **修改** `core/pipeline/pipeline.py` — _run_tasks() 插件循环 + _render() 摔倒绘制
- **修改** `core/alert_manager.py` — fall_detected / fall_potential / fall_recovered 分支
- **修改** `main.py` — build_tasks() 从 plugins.fall_detection 导入
- **修改** `configs/default.yaml` — tasks.fall_detection 完整配置节
- **修改** `requirements/base.txt` — +scipy
- **修改** `requirements/desktop.txt` — +ultralytics +lap
- **删除** `plugins/fall_detection_stub.py` — 空实现替换为真实实现
- **修复** `plugins/fall_engine/backends/postprocess.py` — ONNX 坐标缩放 (pixel space → normalized)
- **修复** FallDetectionTask 匹配策略 — 中心距离替换 IoU (face bbox vs body bbox)

### 原人脸识别项目日志

**v9.4 — 多摄像头架构 (2026-05-15)**
- 新增 `core/camera_pipeline.py` — 单摄像头独立线程 (own tracker/memory/state)
- 新增 `core/multi_camera_manager.py` — 多 CameraPipeline 管理
- 新增 `core/scheduler/global_inference_scheduler.py` — 令牌桶限制同时 detect
- 新增 `configs/cameras.yaml` — 多摄像头 YAML 配置
- 修改 `main.py` — `--multi-camera` + `--camera 0 1` 多源

**v9.3 — 真正边缘化 (2026-05-14)**
- 新增 `core/tracking/iou_tracker.py` — 轻量 IoU 跟踪器 (纯 Python + numpy, 零 boxmot)
- 修复 `main.py` build_tracker() — 根据 tracking.type 选择 (iou→LightweightIoUTracker / bytetrack→MultiObjectTracker)
- 清理 4 个配置文件 — edge_minimal/balanced 仅 iou, desktop 保留 ByteTrack
- 209/209 测试通过

**v9.2 — 边缘部署配置 + 依赖分层 (2026-05-14)**
- 新增 3 个 profile 配置文件: edge_minimal / balanced / desktop
- 新增 `main.py` _deep_merge() 配置级联
- 新增 5 层依赖: requirements/{base,edge_cpu,desktop,jetson,dev}.txt
- 新增 --benchmark 参数 (自动 --no-render --max-frames 300)

**v9.1 — VisionTask 插件接口 (2026-05-14)**
- 新增 `core/interfaces.py` — VisionTask ABC + VisionEvent 数据结构
- 新增 `plugins/fall_detection_stub.py` — 摔倒检测空实现
- 新增 `docs/fall_detection_integration.md` — 13 节融合设计文档
- 修改 `core/pipeline.py` — _run_tasks() 插件循环

**v9.0 — 识别性能优化 (2026-05-14)**
- 新增 `core/face_quality.py` — FaceQualityFilter (尺寸/模糊/长宽比 5 规则)
- 重写 `core/scheduler/recognition_scheduler.py` — v2: identity cooldown + failed backoff
- 重写 `core/workers/recognition_worker.py` — v2: bounded queue + quality-aware submit
- 强化 `core/person/person_manager.py` — EmbeddingCache (LRU 128, 30s TTL)
- 优化 `database/face_db.py` — numpy 向量化 np.dot

**v8.0 — 减法重构 + 行为层 (2026-05-14)**
- 重构 `main.py` — 9 个 build_* 函数 + build_optional_modules()
- 新增 `configs/default.yaml` runtime 段 6 个 enable 开关
- 修改 pipeline.py — 可选模块支持 None + no_render 模式
- 测试重组: core/ / integration/ / manual/ 三层

**v7.x — SCRFD + ByteTrack + 行为层 (2026-05-13)**
- SCRFD 探测器稳定化 (InsightFace, CUDA backend)
- Motion Gate + Frame Scheduler
- TrackMemory (Hungarian 全局匹配 + 锁)
- PersonManager (Re-ID embedding 缓存)
- TrajectoryAnalyzer + BehaviorEngine (MOVING/STATIONARY/LOITERING)
- RegionManager + EventSystem + AlertManager
- 识别调度器 (300帧 cooldown)
- 硬去重 + 重叠/分离触发器

### 原摔倒检测项目日志

**v3.5 — 容错 Runtime (2026-05-15)**
- 新增 `fall_detection/runtime_state/` — 容错状态管理包
- RuntimeStateMachine — 11 状态 + 非法转换 reject
- RuntimePolicy — overload / reconnect / frame drop / degradation
- BackpressureController — 自动帧丢弃 / 队列裁剪 / FPS 降级
- DegradationController — 5 级自动降级
- FaultRecovery — backend/worker/camera 自动恢复
- ResourceManager + RuntimeClock + RuntimeDiagnostics
- `tools/runtime_chaos_test.py` — Chaos 测试

**v3.4 — Runtime Engine (2026-05-15)**
- 新增 `fall_detection/engine/` — 13 文件 Runtime Engine 模块
- RuntimeEngine / RuntimeSession / CameraSession
- SharedFrameBuffer (ring buffer + ref counting)
- EventBus (pub/sub) / RuntimeScheduler / RuntimeWorker
- RuntimeRegistry / RuntimeMetrics / RuntimeHealthMonitor

**v3.3 — 纯 Runtime 重构 (2026-05-14)**
- Runtime 内部彻底禁止 dict, Detection→TrackState→Event 全链路纯对象
- backend infer() 严格返回 list[Detection]
- tracking.py 输入 list[Detection] → 输出 list[TrackState]
- 删除 legacy 兼容层, serializers.py 成唯一 object→dict 出口

**v3.2 — Runtime Core (2026-05-14)**
- 新增 `fall_detection/core/` — AI Monitoring Runtime Core
- Detection / Keypoint / TrackState / Event / FrameContext dataclass
- DetectionPipeline — 6 阶段编排器
- EventRuntime — 事件生成 + cooldown 去重

**v3.1 — 后端抽象 (2026-05-14)**
- 新增 `fall_detection/backends/` — 推理后端抽象层
- UltralyticsBackend / ONNXBackend / BackendFactory
- detector.py 不再 import YOLO，完全 backend 无关

**v3.0 — SDK 化与边缘部署 (2026-05-14)**
- 新增 `fall_detection/` 标准检测包
- FallDetector 类 + JSON 标准化输出
- --headless / --edge CLI 参数

**v2.0 — 追踪 + 降噪 + 误报抑制 (2026-05-13)**
- ByteTracker 集成，Ghost 目标机制
- EMA + Savitzky-Golay 降噪
- 四路检测 + 滑动窗口 + 持续时间确认

**v1.0 — 初始化 (2026-05-11)**
- YOLOv8-Pose 单摄像头实时摔倒检测

---

## 部署指南

### Raspberry Pi 5 / Orange Pi

```bash
pip install -r requirements/edge_cpu.txt
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_s').prepare(ctx_id=-1)"
python register_face.py --name YourName --simple
python main.py --profile edge_minimal --max-frames 100
```

### Jetson Nano / Orin

```bash
pip install -r requirements/jetson.txt
# 推荐 TensorRT，当前 ONNX 过渡
python main.py --profile edge_minimal --device cpu
```

### RK3588 / ARM Edge

```bash
pip install -r requirements/edge_cpu.txt
python main.py --profile edge_minimal --camera 1
```

### Desktop GPU (RTX / GTX)

```bash
pip install -r requirements/desktop.txt
python main.py --profile desktop
```

### Benchmark

```bash
# 自动 --no-render --max-frames 300
python main.py --profile edge_minimal --benchmark

# 自定义帧数
python main.py --profile balanced --benchmark --max-frames 1000
```

### 依赖分层

```
requirements/
├── base.txt          # numpy + pyyaml + opencv-python-headless + scipy
├── edge_cpu.txt      # base + insightface + onnxruntime (CPU)
├── desktop.txt       # base + insightface + onnxruntime-gpu + boxmot + ultralytics + lap
├── jetson.txt        # base + insightface
└── dev.txt           # base + pytest + black + ruff
```

---

## FAQ

<details>
<summary><b>如何切换后端？</b></summary>

编辑 `configs/default.yaml`:
```yaml
tasks:
  fall_detection:
    backend: "onnx"        # ONNX Runtime (边缘推荐, 零 PyTorch)
    backend: "ultralytics"  # Ultralytics YOLO (GPU)
```
</details>

<details>
<summary><b>FPS 太低怎么办？</b></summary>

- 启用 motion gate (默认 ON)
- 设置 `detector.input_size: 320`
- 设置 `tasks.fall_detection.interval: 10`
- 设置 `tasks.fall_detection.input_size: 320`
- 使用 `buffalo_s` 模型
</details>

<details>
<summary><b>两个人显示同一个名字？</b></summary>

增大 `recognition_threshold` 到 0.75-0.8。硬去重确保每注册名仅一框。系统每 300 帧自动重验证。
</details>

<details>
<summary><b>交叉时身份互换？</b></summary>

Hungarian 锁防止交叉时 ID 交换。Re-ID 缓存 (10s TTL) 保留跨丢失/恢复 track 的身份。分离触发器 (0.5s) 在重叠结束后强制重识别。
</details>

<details>
<summary><b>换模型后人脸不识别？</b></summary>

buffalo_l 和 buffalo_s 特征不兼容。需重新注册所有人脸。
</details>

<details>
<summary><b>摄像头打不开？</b></summary>

尝试 `python main.py --camera 0` (内置) 或 `--camera 1` (外接)。Windows: 设置 → 隐私 → 相机 → 允许应用访问。
</details>

<details>
<summary><b>如何在树莓派上运行？</b></summary>

```bash
pip install -r requirements/edge_cpu.txt
python main.py --profile edge_minimal
```
</details>

<details>
<summary><b>摔倒检测不工作？</b></summary>

确认 `configs/default.yaml` 中:
- `tasks.fall_detection.enabled: true`
- `runtime.enable_event_system: true`
- `runtime.enable_alert: true`
- `models/yolov8n-pose.onnx` 文件存在
</details>

<details>
<summary><b>怎么运行测试？</b></summary>

```bash
python tests/run_all_tests.py                    # 全量
python tests/test_fall_detection_integration.py  # 摔倒检测
```
</details>

<details>
<summary><b>InsightFace 编译报错？</b></summary>

先安装 VS C++ Build Tools → "Desktop development with C++"，然后:
```bash
pip install insightface-0.7.3\insightface-0.7.3
```
</details>

---

## License

LYUN License

---

<p align="center">
  <sub>Built for edge AI and real-time vision monitoring.</sub>
</p>
