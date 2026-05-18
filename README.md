# AI 智能监控系统

<p align="center">
  <img src="https://img.shields.io/badge/Version-v10-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/InsightFace-0.7.3-red" alt="InsightFace">
  <img src="https://img.shields.io/badge/SCRFD-500m-cyan" alt="SCRFD">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-ONNX-orange" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/NCNN-ARM_NEON-blue" alt="NCNN">
  <img src="https://img.shields.io/badge/Edge-Zero_BoxMOT-brightgreen" alt="Edge">
  <img src="https://img.shields.io/badge/License-LYUN-lightgrey" alt="License">
</p>

<p align="center">
  <b>统一 AI 智能监控系统 — 人脸识别 (SCRFD + ArcFace) + 摔倒检测 (YOLOv8-Pose) 融合<br>
  PersonManager 唯一追踪源 | 异步推理架构 | NCNN 边缘加速 | 桌面/边缘双部署</b>
</p>

---

## 简介

**AI 智能监控系统** 将人脸识别与摔倒检测融合为单一实时推理管线。

以人脸识别项目 Pipeline 为主体，摔倒检测作为 **VisionTask 插件** 嵌入，**PersonManager** 是唯一追踪数据源 — 人脸和姿态共用 `track_id`，身份与摔倒状态联动告警。

适用场景：
- **智慧养老** — 人脸身份识别 + 摔倒实时告警 ("张三摔倒!" vs "陌生人摔倒!")
- **智能安防** — 人脸门禁 + 区域警戒 + 行为分析
- **医院监护** — 病人移动监测 + 离床告警 + 身份确认
- **边缘 AI** — 树莓派 / Jetson / RK3588 低功耗部署

---

## Demo

```
┌──────────────────────────────────────────────────────────────────┐
│ FPS: 28.5    Persons: 2   Fall: 0 alert(s)    Frame: #1500       │
│ Motion: 2.3(Y) | normal(2f)  Q: 1 | Cache: 3                    │
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

> 绿色框=已知身份 / 灰色框=未知 / 深蓝框=人体 / 金色骨架 / 绿Normal / 橙Warning / 红FALL

---

## 系统架构

```
main.py (统一入口)
    │
    ├─ Camera → MotionGate → FrameScheduler → SCRFD(人脸检测)
    │                                          │
    │                 PersonManager(★唯一追踪源: track_id, bbox, identity)
    │                     │
    │      ┌──────────────┼──────────────┐
    │      ▼              ▼              ▼
    │   Recognition    FallDetection   Behavior
    │   Worker(线程)   Worker(线程)    Engine(可选)
    │   (ArcFace)      (YOLO-Pose)     (MOVING/STATIONARY)
    │      │              │              │
    │      └──────────────┴──────────────┘
    │                     │
    │                     ▼
    │              EventSystem → AlertManager
    │              (fall_detected / stranger_alert)
    │                     │
    │                     ▼
    │              Renderer(统一渲染: 人脸框+骨架+摔倒标签)
```

**每阶段技术：**

| 阶段 | 技术 | 依赖 |
|------|------|------|
| 人脸检测 | SCRFD det_500m (InsightFace) | `insightface` + `onnxruntime` |
| 姿态估计 | YOLOv8n-Pose (ONNX/NCNN) | `onnxruntime` / `ncnn` |
| 追踪 | LightweightIoUTracker / ByteTrack | `boxmot`(桌面) 或 `numpy`(边缘) |
| 身份 | PersonManager + EmbeddingCache LRU | `numpy` |
| 人脸识别 | ArcFace w600k_mbf | `insightface` |
| 摔倒判断 | 四路检测 (几何+物理+侧倒+已倒地) + 滑动窗口 | `numpy` + `scipy` |
| 告警 | EventSystem + AlertManager (冷却去重) | (stdlib) |
| 渲染 | OpenCV (人脸框+骨架+状态标签) | `opencv-python-headless` |

---

## 目录结构

```
ai智能监控（已经融合版）/
├── main.py                        # 统一入口
├── register_face.py               # 人脸注册 CLI
├── README.md                      # 本文件
├── configs/
│   ├── default.yaml               # 统一配置 (face + fall)
│   ├── edge_minimal.yaml          # 320px/CPU/NCNN/interval=15
│   ├── balanced.yaml              # 416px/CPU/ONNX
│   └── desktop.yaml               # 640px/CUDA/ByteTrack
├── core/
│   ├── interfaces.py              # VisionTask ABC + VisionEvent
│   ├── pipeline/pipeline.py       # 主循环 (_run_tasks + fall rendering)
│   ├── person/                    # PersonManager (唯一追踪源)
│   ├── detectors/                 # SCRFD 人脸检测
│   ├── tracking/                  # LightweightIoUTracker / ByteTrack
│   ├── recognition/               # ArcFace 识别
│   ├── scheduler/                 # RecognitionScheduler
│   ├── workers/
│   │   ├── recognition_worker.py  # 异步人脸识别线程
│   │   └── fall_detection_worker.py  # 异步摔倒检测线程 ★
│   ├── rendering/renderer.py      # 统一渲染 (人脸框+骨架+摔倒)
│   ├── event_system.py            # 事件分发
│   ├── alert_manager.py           # 告警 (含 fall 分支)
│   └── ...
├── plugins/
│   ├── fall_detection.py          # FallDetectionTask (VisionTask 插件) ★
│   └── fall_engine/               # 摔倒检测引擎 ★
│       ├── config.py              # 可注入配置
│       ├── fall_logic.py          # 核心摔倒判断 (4路径+滑动窗口)
│       ├── features.py            # 物理特征 (RE/GF/AR/Angle/HD)
│       ├── detection.py           # Detection/Keypoint 数据结构
│       └── backends/
│           ├── factory.py         # create_backend() 工厂
│           ├── base.py            # BaseInferenceBackend ABC
│           ├── ultralytics_backend.py
│           ├── onnx_backend.py
│           ├── ncnn_backend.py    # NCNN ARM NEON 后端 ★
│           └── postprocess.py     # 统一后处理
├── models/
│   ├── scrfd_500m_bnkps.onnx      # SCRFD 人脸检测
│   └── yolov8n-pose.onnx          # YOLOv8-Pose ONNX (12.9 MB)
├── requirements/
│   ├── base.txt                   # numpy + pyyaml + opencv + scipy
│   ├── edge_cpu.txt               # base + insightface + onnxruntime
│   ├── desktop.txt                # base + insightface + onnx-gpu + boxmot + ultralytics
│   ├── jetson.txt                 # base + insightface
│   └── dev.txt                    # base + pytest
├── tests/
│   ├── test_fall_detection_integration.py  # 摔倒检测集成测试 ★
│   └── core/ (17 test modules)
├── database/face_db.py
├── face_db/identities.pkl
└── docs/
    ├── INTEGRATION_GUIDE.md
    └── specs/ (design docs)
```

---

## 功能特性

### 人脸识别
- SCRFD 人脸检测 (InsightFace ONNX, CUDA/CPU)
- Motion Gate 运动门控 (静止帧跳过)
- 自适应帧调度 (1f/2f/5f/15f)
- 多目标追踪 (ByteTrack 桌面 / LightweightIoUTracker 边缘)
- ArcFace 识别 (512d, 余弦相似度 ≥0.70)
- 异步识别 Worker (守护线程, 非阻塞)
- 人脸质量过滤 (5 规则: 尺寸/模糊/长宽比/边界/综合)
- Embedding 缓存 (LRU 128, 30s TTL)
- 硬去重 (注册用户 1 框上限)

### 摔倒检测
- YOLOv8-Pose 姿态估计 (17 关键点 COCO)
- 四路摔倒判断 (几何规则 + 物理特征 RE/GF + 侧倒 + 已倒地)
- 滑动窗口确认 (10 帧窗口, 50% 触发比, 连续触发)
- 持续时间确认 (3.5s → FALL)
- 回弹/恢复检测
- EMA + Savitzky-Golay 双平滑
- 异步 FallDetectionWorker (守护线程, body-to-body IoU 追踪)
- 三后端: ONNX / Ultralytics / NCNN (ARM NEON)

### 融合增强
- PersonManager 唯一追踪 (face + pose 共用 track_id)
- 摔倒+身份联动告警 ("张三摔倒!" vs "陌生人摔倒!")
- 统一渲染 (人脸框 + 人体框 + 骨架 + 摔倒状态)
- 统一配置 (单 YAML, tasks.fall_detection 独立节)
- NCNN ARM NEON 边缘加速

---

## 快速开始

### 环境要求

| 软件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 全部 |
| VS C++ Build Tools | 2022 | Windows InsightFace 编译 |
| NVIDIA GPU | GTX 1060 6GB+ | desktop profile |
| CUDA Toolkit | 11.8+ | desktop profile |

### 安装

```bash
# 桌面 GPU
pip install -r requirements/desktop.txt

# 边缘 CPU (树莓派)
pip install -r requirements/edge_cpu.txt
```

### 导出 ONNX 模型 (首次)

```bash
python -c "from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt').export(format='onnx', imgsz=640)"
```

### 注册人脸

```bash
python register_face.py --name YourName --simple
```

### 运行

```bash
# 人脸识别 + 摔倒检测
python main.py

# GPU 全功能
python main.py --profile desktop

# 边缘低功耗 (320px, NCNN, interval=15)
python main.py --profile edge_minimal

# Benchmark (300帧自动退出)
python main.py --benchmark

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
# 人脸检测
detector:
  model_name: "buffalo_s"
  input_size: 640
  conf_threshold: 0.5
  detection_interval: 2
  device: cuda

# 人脸识别
recognition:
  recognition_threshold: 0.70
  recognition_cooldown: 300

# 摔倒检测
tasks:
  fall_detection:
    enabled: true
    backend: "onnx"                  # onnx / ncnn / ultralytics
    model_path: "models/yolov8n-pose.onnx"
    device: "auto"
    interval: 5                      # 桌面每5帧, 边缘15帧
    input_size: 640                  # 桌面640, 边缘320
    confidence_threshold: 0.5
    fall:
      horizontal_ar_threshold: 0.6
      angle_threshold: 120
      min_fall_pose_duration: 3.5
      window_size: 10
      window_trigger_ratio: 0.5
      min_consecutive_triggers: 3

# 运行时
runtime:
  enable_event_system: true          # 摔倒告警需要
  enable_alert: true
```

### 边缘化配置

| 参数 | 桌面 | 边缘 | 说明 |
|------|------|------|------|
| `backend` | onnx | ncnn | ARM NEON 加速 |
| `input_size` | 640 | 320 | 像素数 1/4 |
| `interval` | 5 | 15 | 降低推理频率 |
| `confidence_threshold` | 0.5 | 0.4 | 补偿小分辨率 |

---

## 性能

### RTX 4060 Laptop (人脸+摔倒同时运行)

| 阶段 | 延迟 | 频率 |
|------|------|------|
| SCRFD 640px | ~15ms | 每 2 帧 |
| YOLO ONNX 640px | ~42ms | 每 5 帧 |
| ArcFace | ~5ms | 按需 (300f 冷却) |
| 追踪+渲染 | ~3ms | 每帧 |
| **总 (平均)** | **~15-30ms** | **~30-65 FPS** |

### 树莓派 5 (NCNN, 320px)

| 阶段 | 延迟 | 频率 |
|------|------|------|
| SCRFD 320px | ~40ms | 每 4 帧 |
| YOLO NCNN 320px | ~35ms | 每 15 帧 |
| **总 (预计)** | | **~20-25 FPS** |

---

## 更新日志

### v10 — 项目融合 (2026-05-18)

- **融合** 人脸识别 + 摔倒检测为统一系统
- **新增** `plugins/fall_engine/` (fall_logic.py + features.py + backends/ + detection.py)
- **新增** `plugins/fall_detection.py` — FallDetectionTask (VisionTask 插件)
- **新增** `core/workers/fall_detection_worker.py` — 异步摔倒检测线程 (body-to-body IoU 追踪)
- **新增** `plugins/fall_engine/backends/ncnn_backend.py` — NCNN ARM NEON 后端
- **新增** `models/yolov8n-pose.onnx` (12.9 MB)
- **新增** `core/rendering/renderer.py` — 骨架 + 人体框 + 摔倒状态
- **修改** `core/pipeline/pipeline.py` — _run_tasks() + fall rendering
- **修改** `core/alert_manager.py` — fall_detected / fall_potential / fall_recovered
- **修改** `configs/default.yaml` + `edge_minimal.yaml` — fall 配置节
- **修改** `requirements/base.txt` (+scipy), `desktop.txt` (+ultralytics +lap)
- **修复** aspect_ratio 方向 (h/w 非 w/h) — Path 1 几何检测恢复
- **修复** ONNX 后处理坐标归一化
- **测试** test_fall_detection_integration.py (5/5 通过)

### v9.1-v9.4 — 人脸识别 (原项目)

- **v9.4**: 多摄像头架构 (MultiCameraManager, GlobalInferenceScheduler)
- **v9.3**: 真正边缘化 (LightweightIoUTracker, 零 boxmot)
- **v9.2**: 边缘部署配置 + 5层依赖分层
- **v9.1**: VisionTask 插件接口

---

## License

LYUN License
