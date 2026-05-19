# AI Smart Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Version-v10-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/InsightFace-0.7.3-red" alt="InsightFace">
  <img src="https://img.shields.io/badge/SCRFD-500m-cyan" alt="SCRFD">
  <img src="https://img.shields.io/badge/YOLOv8--Pose-ONNX-orange?logo=yolo" alt="YOLOv8-Pose">
  <img src="https://img.shields.io/badge/NCNN-ARM_NEON-blue" alt="NCNN">
  <img src="https://img.shields.io/badge/Edge-Zero_BoxMOT-brightgreen" alt="Edge">
  <img src="https://img.shields.io/badge/License-LYUN-lightgrey" alt="License">
</p>

<p align="center">
  <b>A production-grade real-time edge AI monitoring system — SCRFD detection + ByteTrack tracking + ArcFace recognition<br>+ YOLOv8-Pose fall detection + NCNN ARM NEON acceleration + behavior analysis + GPU-accelerated async inference.</b>
</p>

---

## Introduction

**AI Smart Monitoring System** is a production-grade real-time AI monitoring pipeline that fuses **face recognition** and **fall detection** into a single unified system for edge deployment.

It combines state-of-the-art models — SCRFD for lightweight face detection, ByteTrack for multi-object tracking, InsightFace ArcFace for face recognition, YOLOv8-Pose for human pose estimation, a four-path fall detection algorithm, NCNN ARM NEON backend, and a full behavior analysis layer — into a single optimized pipeline running entirely on local hardware at real-time speeds.

Designed for scenarios where cloud dependency, latency, or subscription cost is unacceptable:

- **Smart Security** — face-based access control, visitor logging, zone alerts
- **Elderly Care** — real-time fall detection with identity linkage, nursing home monitoring
- **Smart Hospital** — patient movement monitoring, bed-exit alerts, identity confirmation
- **Retail Analytics** — customer counting, demographic analysis
- **Smart Home** — family member recognition, automation triggers, in-home fall alerts
- **Edge AI Research** — multi-model pipeline benchmarking, ONNX vs NCNN comparison

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
│   [EVENT] Byron entered server_room                              │
│   [EVENT] FALL DETECTED track=5 conf=0.60                        │
│   Press Q to quit                                                │
└──────────────────────────────────────────────────────────────────┘
```

> *Face bounding-box (green=identified / gray=unknown) + body bounding-box (dark blue) + 17-keypoint skeleton (gold) + fall status (green Normal / orange Warning / red FALL)*

---

## Features

### Core Pipeline

- [x] **SCRFD Face Detection** — InsightFace det_500m ONNX, CUDA GPU inference, 5-point landmarks
- [x] **Motion Gate** — frame-difference motion detection (threshold 2.0), skips idle frames
- [x] **Adaptive Frame Scheduler** — dynamic detection intervals (fast=1f / normal=2f / slow=5f / force=15f)
- [x] **ByteTrack Multi-Object Tracking** — Kalman filter + Hungarian global matching, stable IDs (desktop)
- [x] **Lightweight IoU Tracker** — pure Python + numpy, zero boxmot dependency, edge-optimized (edge_minimal/balanced)
- [x] **TrackMemory** — long-term trajectory memory with direction penalty, lock mechanism, size weight
- [x] **Track Reassociation** — three-tier lost-track recovery (IoU → spatial proximity → no match)
- [x] **ArcFace Recognition** — 512-dim embeddings via InsightFace 0.7.3 (buffalo_s)
- [x] **Async Recognition Worker** — daemon thread, non-blocking submit + non-blocking collect
- [x] **Smart Scheduler** — priority: new tracks → cooldown-expired → re-verify identified
- [x] **Face Re-ID Cache** — embedding cache (10s TTL) preserves identity across ID switches
- [x] **Hard Dedup** — one registered-name box max; duplicates reset with 600-frame cooldown
- [x] **PersonManager** — sole tracking source, unified track_id for face + pose

### Fall Detection (v10)

- [x] **YOLOv8-Pose** — 17-keypoint human pose estimation (COCO format)
- [x] **Four-Path Detection** — Geometry (AR+angle) + Physics (RE/GF) + Side-fall (AR+head descent) + Already-down
- [x] **Sliding Window** — 10-frame trigger ratio (50%), consecutive trigger confirmation
- [x] **Duration Confirmation** — 3.5s persistence before confirmed FALL
- [x] **Rebound Detection** — head rebound 15% body height → cancel fall
- [x] **Recovery Detection** — AR recovery to 70% baseline auto-reset
- [x] **EMA + Savitzky-Golay** — dual smoothing for noise reduction (RE / GF / Head Descent)
- [x] **Torso Inclination** — hip→shoulder vector angle, monitoring-view adaptive
- [x] **Async FallDetectionWorker** — daemon thread, body-to-body IoU tracking, non-blocking submit/poll
- [x] **Three Backends** — ONNX (CPU/CUDA) / Ultralytics (PyTorch) / NCNN (ARM NEON, Raspberry Pi)
- [x] **Backend Factory** — `create_backend("onnx"|"ncnn"|"ultralytics")` one-click switch
- [x] **Ghost Tracking** — lost-track fall state inheritance, fallen ghosts persist 30s for re-identification
- [x] **Cross-Camera Matching** — HSV histogram matching for dual-camera fall confirmation
- [x] **Fall+Identity Linkage** — "Byron fell!" vs "Stranger fell!" differentiated alerts

### Behavior Analysis Layer

- [x] **Trajectory Analyzer** — per-track speed, direction, stationary frame accumulation
- [x] **Behavior Engine** — state machine: MOVING / STATIONARY / LOITERING / DISAPPEARED
- [x] **Region Manager** — zone system with point-in-polygon entry/leave tracking
- [x] **Event System** — unified event emitter (face events + fall events)
- [x] **Alert Manager** — cooldown-based alert dedup (30s per alert type, fall-aware)

### Trigger System

- [x] **Out-of-frame trigger** — force re-recognize when track exits frame boundary
- [x] **Overlap trigger** — reset identity immediately when IoU > 0.3
- [x] **Separation trigger** — unlock Hungarian lock + force re-recognize after 0.5s separation
- [x] **Hard dedup trigger** — duplicate registered names reset with 600-frame cooldown

### Recognition Optimization (v9)

- [x] **Face Quality Filter** — 5-rule pre-screening (size/blur/aspect/bounds), rejects ~70% low-quality faces
- [x] **Identity Cooldown** — recognized persons get 600-frame cooldown (vs 300 for unknown)
- [x] **Failed Backoff** — per-track exponential backoff (90f × fail_count), caps at 20 attempts
- [x] **Queue Pressure Gate** — when worker queue ≥3, only new tracks allowed
- [x] **Embedding Cache** — LRU cache (128 entries, 30s TTL) avoids re-recognition after track recovery
- [x] **Vectorized DB Search** — numpy `np.dot(matrix, query)` O(N×512) single-operation cosine similarity

### VisionTask Plugins (v9)

- [x] **VisionTask Interface** — abstract base class for pluggable vision tasks (should_run + run)
- [x] **VisionEvent** — unified event data structure (event_type, track_id, confidence, payload)
- [x] **FallDetectionTask** — real implementation with YOLOv8-Pose + evaluate_fall + async worker
- [x] **Pipeline Integration** — tasks list with per-task try/except + PerformanceMonitor timing

### Engineering

- [x] **3 Deployment Profiles** — edge_minimal(320px/CPU/NCNN) / balanced(416px/CPU) / desktop(640px/CUDA/ByteTrack)
- [x] **Layered Dependencies** — `requirements/{base,edge_cpu,desktop,jetson,dev}.txt`
- [x] **Per-Stage Latency** — SCRFD pre/infer/post + YOLO fall timing, rolling average
- [x] **GPU/CPU Dual Mode** — auto-detect CUDAExecutionProvider, seamless CPU fallback
- [x] **YAML Configuration** — all tunable parameters with profile cascade (default → profile → CLI args)
- [x] **CLI Face Manager** — register / list / remove faces via command line
- [x] **Thread-Safe Architecture** — main thread owns PersonManager; workers process in daemon threads
- [x] **Metrics Logger** — counters, gauges, runtime tracking, summary reports

### Coming Soon

- [ ] TensorRT acceleration (YOLOv8-Pose 2-3x faster)
- [ ] Web Dashboard — FastAPI + WebSocket
- [ ] Multi-camera RTSP streaming
- [ ] OpenVINO backend
- [ ] Mobile app push notifications

---

## System Architecture

### Single-Camera Pipeline (v10 — Fused)

```
┌─────────────────────────────── Main Thread ───────────────────────────────────┐
│                                                                               │
│  Camera ──▶ MotionGate ──▶ FrameScheduler ──▶ SCRFD(降频)                     │
│  (cv2)      (diff>2.0?)     (1/2/5/15fr)       InsightFace                   │
│                     │              │              │                            │
│                     ▼              ▼              ▼                            │
│              [skip frame]   [skip frame]   ByteTrack + Hungarian              │
│                                                    │                          │
│                                                    ▼                          │
│                                             PersonManager                     │
│                                             (★唯一追踪源)                     │
│                                             ·track_id + identity              │
│                                             ·embedding cache                  │
│                                                    │                          │
│              ┌─────────────────────────────────────┤                          │
│              ▼                                     ▼                          │
│   RecognitionWorker(thread)              FallDetectionWorker(thread)          │
│   ·submit(crop) 非阻塞                   ·submit(frame) 非阻塞                │
│   ·ArcFace → FaceDB                      ·YOLO-Pose → evaluate_fall()        │
│   ·poll_results()                        ·body-to-body IoU tracking           │
│              │                                     │                          │
│              └─────────────────────────────────────┤                          │
│                                                    ▼                          │
│                                              EventSystem                      │
│                                              ·fall_detected                   │
│                                              ·stranger_alert                  │
│                                                    │                          │
│                                                    ▼                          │
│                                              AlertManager                     │
│                                              ·30s cooldown                    │
│                                              ·身份+摔倒联动                   │
│                                                    │                          │
│                                                    ▼                          │
│                                               Renderer                        │
│                                              ·人脸框+身份标签                 │
│                                              ·人体框(深蓝)+骨架(金色)         │
│                                              ·摔倒状态(绿Normal/橙Warn/红FALL) │
│                                                    │                          │
│                                                    ▼                          │
│                                              cv2.imshow()                     │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Tech per stage (v10):**

| Stage | Technology | Dependency |
|-------|-----------|------------|
| Capture | OpenCV VideoCapture | `opencv-python-headless` |
| Motion | Frame-difference | `numpy` |
| Schedule | Adaptive FrameScheduler | (stdlib) |
| Face Detect | SCRFD det_500m (InsightFace) | `insightface` + `onnxruntime` |
| Pose Detect | YOLOv8n-Pose | `onnxruntime` or `ncnn` or `ultralytics` |
| Track | ByteTrack or LightweightIoUTracker | `boxmot` (desktop) or `numpy` (edge) |
| Memory | TrackMemory Hungarian | `numpy` |
| Person | PersonManager + EmbeddingCache | `numpy` |
| Quality | FaceQualityFilter (5 rules) | `numpy` + `cv2` |
| Recognize | ArcFace w600k_mbf | `insightface` |
| Fall Judge | 4-path logic + sliding window | `numpy` + `scipy` |
| Event | EventSystem + AlertManager | (stdlib) |
| Render | OpenCV (face + body + skeleton + status) | `opencv-python-headless` |

### Multi-Camera Pipeline (v9.4)

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
│  │ #cam0        │ │ #cam1        │ │ #cam2        │        │
│  │ (thread)     │ │ (thread)     │ │ (thread)     │        │
│  │              │ │              │ │              │        │
│  │ OWN: tracker │ │ OWN: tracker │ │ OWN: tracker │        │
│  │      memory  │ │      memory  │ │      memory  │        │
│  │      person  │ │      person  │ │      person  │        │
│  │      metrics │ │      metrics │ │      metrics │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**Key difference from single-camera:**

| Resource | Single-Camera | Multi-Camera |
|----------|--------------|--------------|
| Tracker | 1 shared | 1 per camera |
| TrackMemory | 1 shared | 1 per camera |
| PersonManager | 1 shared | 1 per camera |
| Track IDs | global namespace | per-camera isolated |
| RecognitionWorker | own | **shared (1 instance)** |
| FaceDatabase | own | **shared (1 instance)** |
| FallDetectionWorker | own | 1 per camera (planned) |
| Inference | sequential | **GlobalInferenceScheduler (max 2 concurrent)** |
| Threads | 1 main + 2 workers | 1 per camera + 1 shared worker |

---

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Async recognition in worker thread | ArcFace inference (5-8ms) never blocks video rendering |
| Async fall detection in worker thread | YOLO-Pose inference (42ms) never blocks main loop; cached results rendered |
| PersonManager as sole tracking source | Single track_id shared by face and pose; eliminates dual-track sync issues |
| Body-to-body IoU tracking (fall) | Independent of face tracker; stable fall_track_id across face ID resets |
| Motion-gated detection | Skips SCRFD on static frames, reduces GPU load significantly |
| Adaptive detection intervals | fast=1f (motion), normal=2f, slow=5f (idle), force=15f (max skip) |
| Hungarian global matching with lock | Locked confirmed matches prevent ID swaps during crossings |
| No velocity prediction | Removed due to variable intervals; rely on center distance + size weight |
| Hard dedup before render | One registered-name box max; duplicates reset with 600fr cooldown |
| Face Re-ID cache (10s TTL) | Lost tracks' embeddings cached; new tracks auto-match to restore identity |
| Separation trigger (0.5s) | Unlock Hungarian locks + force re-recognition after overlapping boxes separate |
| Single-task worker queue | Prevents multi-face GPU memory contention |
| 300-frame recognition cooldown | 10-30x fewer ArcFace calls vs per-frame recognition |
| Fall event dedup (state-change only) | Prevents event flood on sticky fall states |
| Thread-safe by design | Main thread owns PersonManager; workers own internal tracking state |
| Config injection (not global import) | `fall_engine` receives config at runtime, enables profile switching |

---

## Tech Stack

### Face Recognition

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Face Detection | SCRFD det_500m (InsightFace) | 0.7.3 | Lightweight face detection + 5-point landmarks |
| Object Tracking (desktop) | ByteTrack (BoxMOT) | 18.x | Multi-object Kalman tracking |
| Object Tracking (edge) | LightweightIoUTracker | — | Greedy IoU, zero deps, edge CPU |
| Long-term Memory | TrackMemory (Hungarian) | — | Global optimal assignment + lock |
| Track Recovery | TrackReassociation | — | IoU → spatial → none matching |
| Face Recognition | ArcFace ONNX (InsightFace) | 0.7.3 | 512-dim embedding extraction |
| Recognition Model | buffalo_s (w600k_mbf) | — | Lightweight mobile-face backbone |
| Face Quality | FaceQualityFilter | — | Size/blur/aspect 5-rule pre-screening |
| Embedding Cache | LRU OrderedDict | — | track_id→embedding, TTL+容量限制 |

### Fall Detection

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Pose Estimation | YOLOv8-Pose (nano) | 8.x | 17-keypoint pose estimation |
| ONNX Infer | ONNX Runtime | ≥ 1.15 | CPU/CUDA backend |
| NCNN Infer | NCNN (Tencent) | latest | ARM NEON native acceleration (Pi) |
| Ultralytics Infer | Ultralytics YOLO | ≥ 8.0 | PyTorch backend |
| Backend Factory | create_backend() | — | onnx / ncnn / ultralytics switch |
| Fall Judge | 4-path logic + sliding window | — | Geometry + Physics + Side-fall + Already-down |
| Physical Features | EMA + Savitzky-Golay | — | RE / GF / HD with dual smoothing |
| Tracking | Body-to-body IoU (Worker) | — | Stable fall_track_id, independent of face |

### Shared

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Motion Detection | Motion Gate (frame-diff) | — | Pixel-level motion gating |
| Frame Scheduling | Adaptive Scheduler | — | Dynamic detection intervals |
| Behavior Analysis | TrajectoryAnalyzer + BehaviorEngine | — | MOVING / STATIONARY / LOITERING |
| Zone System | RegionManager (point-in-polygon) | — | Zone entry/leave tracking |
| Event System | EventSystem + AlertManager | — | Unified events + 30s cooldown |
| VisionTask Plugin | VisionTask ABC + VisionEvent | — | Pluggable extension tasks |
| Config | YAML | ≥ 6.0 | Profiles (edge_minimal/balanced/desktop) |
| Image Processing | OpenCV | ≥ 4.8 | Capture, display, rendering |
| Numerical | NumPy + SciPy | ≥ 1.24 / ≥ 1.10 | Matrix ops + signal smoothing |
| Language | Python | 3.12 | Application logic |

---

## Project Structure

```
project/
│
├── main.py                              # Application entry point
├── register_face.py                     # Face registration CLI
├── requirements.txt                     # → requirements/desktop.txt (backward compat)
├── README.md                            # This document
│
├── configs/
│   ├── default.yaml                     # Unified config (face + fall)
│   ├── edge_minimal.yaml                # 320px / CPU / NCNN / interval=15
│   ├── balanced.yaml                    # 416px / CPU / ONNX
│   ├── desktop.yaml                     # 640px / CUDA / ByteTrack
│   └── cameras.yaml                     # Multi-camera config
│
├── requirements/
│   ├── base.txt                         # numpy + pyyaml + opencv + scipy
│   ├── edge_cpu.txt                     # base + insightface + onnxruntime (CPU)
│   ├── desktop.txt                      # base + insightface + onnxruntime-gpu + boxmot + ultralytics
│   ├── jetson.txt                       # base + insightface (TensorRT recommended)
│   └── dev.txt                          # base + pytest + black + ruff
│
├── core/
│   ├── interfaces.py                    # VisionTask ABC + VisionEvent
│   ├── face_quality.py                  # Face quality filter (size/blur/aspect)
│   ├── camera/                          # Camera capture
│   ├── detectors/                       # SCRFD face detector (InsightFace)
│   ├── recognition/                     # InsightFace ArcFace wrapper
│   ├── tracking/                        # ByteTrack + LightweightIoUTracker
│   ├── person/                          # Person data model + PersonManager
│   ├── pipeline/                        # Main loop orchestrator (v9)
│   ├── rendering/                       # Frame overlay (face+body+skeleton+status)
│   ├── scheduler/                       # Recognition priority scheduler + GlobalInference
│   ├── workers/
│   │   ├── recognition_worker.py        # Async face recognition worker
│   │   └── fall_detection_worker.py     # ★ Async fall detection worker
│   ├── track_memory.py                  # Long-term trajectory memory (Hungarian)
│   ├── track_reassociation.py           # Lost-track recovery (IoU/spatial)
│   ├── frame_scheduler.py               # Adaptive detection interval scheduler
│   ├── trajectory_analyzer.py           # Per-track speed/direction analysis
│   ├── behavior_engine.py               # Behavior state machine
│   ├── region_manager.py                # Zone system (point-in-polygon)
│   ├── event_system.py                  # Unified event emitter
│   ├── alert_manager.py                 # Cooldown-based alert dedup (fall-aware)
│   ├── multi_camera_manager.py          # Multi-camera orchestration
│   ├── camera_pipeline.py               # Per-camera processing thread
│   ├── cross_camera.py                  # Cross-camera identity matching
│   └── metrics_logger.py                # Unified metrics (counters/gauges)
│
├── plugins/
│   ├── __init__.py                      # Plugin package
│   ├── fall_detection.py                # ★ FallDetectionTask (VisionTask plugin)
│   └── fall_engine/                     # ★ Fall detection engine
│       ├── __init__.py
│       ├── config.py                    # Runtime config injection
│       ├── fall_logic.py                # Core fall judgment (4-path + sliding window)
│       ├── features.py                  # Physical features (RE/GF/AR/Angle/HD)
│       ├── detection.py                 # Detection / Keypoint dataclass
│       └── backends/
│           ├── base.py                  # BaseInferenceBackend ABC
│           ├── factory.py               # create_backend() factory (onnx/ncnn/ultralytics)
│           ├── onnx_backend.py           # ONNX Runtime backend (CPU/CUDA)
│           ├── ncnn_backend.py           # NCNN backend (ARM NEON)
│           ├── ultralytics_backend.py    # Ultralytics YOLO backend (PyTorch)
│           └── postprocess.py           # Unified postprocessing (raw → Detection)
│
├── database/
│   └── face_db.py                       # Pickle-based identity database (vectorized search)
│
├── utils/
│   ├── motion_gate.py                   # Frame-difference motion detection
│   ├── fps.py                           # FPS counter
│   ├── config_loader.py                 # YAML config loader
│   └── performance_monitor.py           # Per-stage latency + recognition counters
│
├── models/
│   ├── scrfd_500m_bnkps.onnx            # SCRFD face detection model
│   └── yolov8n-pose.onnx                # YOLOv8-Pose ONNX model (12.9 MB)
│
└── face_db/
    └── identities.pkl                   # Registered face embeddings
```

---

## Installation

### Prerequisites

| Software | Version | Check | Required For |
|----------|---------|-------|-------------|
| Python | 3.10+ | `python --version` | All profiles |
| VS C++ Build Tools | 2022 | Windows only | InsightFace compilation |
| NVIDIA GPU | GTX 1060 6GB+ | `nvidia-smi` | desktop profile only |
| CUDA Toolkit | 11.8+ | `nvcc --version` | desktop profile only |

> **edge_minimal / balanced 模式不需要 GPU**，仅需 Python + VS Build Tools。

### Step-by-Step

**1. Download VS Build Tools (Windows)**

```
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```
Select **"Desktop development with C++"** → Install.

**2. Install dependencies**

```bash
# Desktop GPU (推荐)
pip install -r requirements/desktop.txt

# Edge CPU (树莓派等)
pip install -r requirements/edge_cpu.txt

# Raspberry Pi with NCNN
pip install ncnn
pip install -r requirements/edge_cpu.txt
```

**3. Compile insightface 0.7.3**

```bash
pip install insightface-0.7.3\insightface-0.7.3
```

**4. Download buffalo_s model (optional)**

If auto-download fails, manually get:

```
https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip
```

Extract to: `C:\Users\<name>\.insightface\models\buffalo_s\`

**5. Export ONNX model (for fall detection, first time only)**

```bash
python -c "from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt').export(format='onnx', imgsz=640)"
```

**6. Register your face**

```bash
python register_face.py --name YourName --simple
```

**7. Run**

```bash
python main.py
```

---

## Usage

### Start

```bash
python main.py --profile desktop               # GPU 全功能 (人脸+摔倒)
python main.py --profile edge_minimal          # 边缘最低功耗 (320px, NCNN)
python main.py --profile balanced              # CPU 平衡模式
python main.py --camera 0                      # 内置摄像头
python main.py --camera 1                      # 手机/USB摄像头
python main.py --device cpu                    # 强制CPU
python main.py --benchmark                     # 自动无渲染 + 300帧测试
```

### Toggle Fall Detection

```yaml
# configs/default.yaml
tasks:
  fall_detection:
    enabled: true    # 开启摔倒检测
    enabled: false   # 关闭 (仅人脸识别)
```

### Manage Faces

```bash
python register_face.py --name Alice --simple     # Register (SPACE capture)
python register_face.py --name Alice --image a.jpg # From photo
python register_face.py --list                     # List all
python register_face.py --remove --name Alice      # Delete
```

### Key Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `ESC` | Cancel registration |

### Display Overlay

| Element | Meaning |
|---------|---------|
| `FPS: 28.5` | Real-time frames per second |
| `Persons: 2` | Stable tracked persons |
| `Fall: 0 alert(s)` | Active fall alerts count |
| `Motion: ON` | Motion gate status |
| `Frame: #1500` | Current frame number |
| `Q: 1` | Recognition queue depth |
| `Cache: 3` | Embedding cache hits |
| 🟢 Face Box | Identified person (green) |
| ⚪ Face Box | Unknown person (gray) |
| 🔵 Body Box | YOLO body bbox (dark blue) |
| 🟡 Skeleton | 17-keypoint pose (gold) |
| 🟢 Normal | Normal posture |
| 🟠 Warning | Potential fall detected |
| 🔴 FALL | Confirmed fall alert |
| `MOVING/STATIONARY` | Per-person behavior state |

---

## Configuration

### Profiles (v10)

Three pre-configured profiles for different hardware tiers:

| Profile | Face Input | Fall Input | Fall Backend | Fall Interval | Tracking | GPU | Render |
|---------|-----------|-----------|-------------|---------------|----------|-----|--------|
| `edge_minimal` | 320px | 320px | ncnn | every 15f | IoU | CPU | no |
| `balanced` | 416px | 416px | onnx | every 10f | IoU | CPU | yes |
| `desktop` | 640px | 640px | onnx | every 5f | ByteTrack | CUDA | yes |

**Priority: CLI args > profile config > default.yaml**

```bash
python main.py --profile edge_minimal              # 边缘最低功耗
python main.py --profile balanced                  # 平衡模式
python main.py --profile desktop                   # 桌面GPU全功能
python main.py --profile edge_minimal --device cuda # profile + CLI覆盖
python main.py --config configs/edge_minimal.yaml   # 直接指定配置文件
```

Profile files: `configs/edge_minimal.yaml`, `configs/balanced.yaml`, `configs/desktop.yaml`

### Why edge_minimal defaults

- **no onnxruntime-gpu**: 边缘设备无 NVIDIA GPU 或显存不足
- **no boxmot (ByteTrack)**: 减少第三方依赖，IoU 跟踪对人脸场景足够
- **render=false**: 减少 SDL/OpenCV 显示开销，仅 benchmark
- **input_size=320**: SCRFD 320px 精度损失 <5%，速度提升 ~3x
- **YOLO input_size=320**: YOLO 320px 推理 ~35ms (NCNN) vs 100ms (ONNX)
- **YOLO interval=15**: 每15帧推理一次，摊销延迟
- **behavior modules OFF**: 减少每帧 100+ 次规则计算
- **NCNN backend**: ARM NEON 原生加速，比 ONNX 快 2-3x

### Parameters

All settings in `configs/default.yaml`:

```yaml
camera:
  index: 0
  width: 640
  height: 480

detector:
  model_name: "buffalo_s"
  input_size: 640
  conf_threshold: 0.5
  detection_interval: 2
  device: cuda

recognition:
  model_name: "buffalo_s"
  recognition_threshold: 0.70
  recognition_cooldown: 300
  min_face_size: 48
  blur_threshold: 80
  max_queue_size: 4

tracking:
  type: iou                  # iou (edge) / bytetrack (desktop)
  iou_threshold: 0.3
  max_lost: 15
  min_hits: 2

# === Fall Detection (v10) ===
tasks:
  fall_detection:
    enabled: true
    model_path: "models/yolov8n-pose.onnx"
    device: "auto"            # auto / cpu / cuda
    backend: "onnx"           # onnx / ncnn / ultralytics
    interval: 5               # desktop=5, edge=15
    input_size: 640           # desktop=640, edge=320
    confidence_threshold: 0.5

    fall:
      horizontal_ar_threshold: 0.6
      angle_threshold: 120
      torso_inclination_threshold: 65
      min_fall_pose_duration: 3.5
      window_size: 10
      window_trigger_ratio: 0.5
      min_consecutive_triggers: 3

runtime:
  mode: edge_minimal
  use_gpu: true
  enable_behavior: false
  enable_event_system: true     # fall alerts require this
  enable_alert: true
  enable_reassociation: false
  enable_region: false

motion:
  threshold: 2.0
  history: 0
  force_interval: 15

behavior:
  stationary_threshold: 60
  loitering_threshold: 300

pipeline:
  window_name: "Vision AI"
  quit_key: "q"
  worker_queue_size: 8
```

### Quick Tuning

```
Goal                    │ Settings
────────────────────────┼─────────────────────────────────────────────────
Maximum FPS             │ motion enabled, detection interval: slow(5f), fall interval: 15
Best Face Accuracy      │ conf_threshold: 0.3, recognition_threshold: 0.8
Best Fall Accuracy      │ fall interval: 1, input_size: 640, enable_roi: true
Low VRAM GPU            │ motion enabled, detection interval: normal(2f)
CPU Only                │ --device cpu, motion enabled, detect slow, fall backend: onnx
Edge (Raspberry Pi)     │ profile edge_minimal, fall backend: ncnn, input_size: 320
Reduced Fall False Pos  │ increase min_fall_pose_duration to 5.0
Faster Fall Detection   │ decrease min_fall_pose_duration, increase window_trigger_ratio
More Alerts             │ decrease alert cooldown, add more zones
```

---

## Performance

### Desktop — RTX 4060 Laptop GPU (Face + Fall v10)

| Stage | Latency | Frequency |
|-------|---------|-----------|
| Camera | 2ms | Every frame |
| Motion Gate | <1ms | Every frame |
| SCRFD Detection (640px) | ~15ms | Adaptive (1-15 frames) |
| YOLO ONNX (640px) | ~42ms | Every 5 frames |
| ByteTrack + Hungarian | ~1ms | Every detect frame |
| Recognition (buffalo_s) | 5ms | On-demand (300fr cooldown) |
| Behavior Analysis | <1ms | Every frame |
| Render (face+body+skeleton) | ~3ms | Every frame |
| **Total (avg)** | **~15-30ms** | **~30-65 FPS** |

### Edge — Raspberry Pi 5 (NCNN, 320px, interval=15, estimated)

| Stage | Latency | Frequency |
|-------|---------|-----------|
| SCRFD 320px | ~40ms | Every 4 frames |
| YOLO NCNN 320px | ~35ms | Every 15 frames |
| Render | ~3ms | Every frame |
| **Total (avg)** | **~15ms** | **~20-25 FPS** |

### Detection Frequency Impact

| Mode | Interval | GPU Load | Latency (avg) |
|------|----------|----------|---------------|
| fast | Every frame | 100% | ~20ms |
| normal | Every 2 frames | 50% | ~12ms |
| slow | Every 5 frames | 20% | ~8ms |
| motion-gated | Adaptive | ~30% | ~6-10ms |

### Multi-Person Scaling

| People | Recognition | Fall Detection | FPS |
|--------|-------------|---------------|-----|
| 1 | Light | Light (1 track) | ≈ camera max |
| 2-3 | Light (queued) | Light (2-3 tracks) | ≈ camera max |
| 4-6 | Moderate | Moderate | slight drop |
| 7+ | Queue fill | Capped at 10 tracks | increase cooldown |

---

---

## Deployment Guide

### Raspberry Pi 5 / Orange Pi

```bash
# 安装边缘依赖 (无GPU包)
pip install ncnn
pip install -r requirements/edge_cpu.txt

# 下载模型
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_s').prepare(ctx_id=-1)"

# 注册人脸
python register_face.py --name YourName --simple

# 运行 (边缘配置: NCNN, 320px, interval=15)
python main.py --profile edge_minimal --max-frames 100
```

### Jetson Nano / Orin

```bash
# Jetson 推荐 TensorRT，当前使用 ONNX 过渡
pip install -r requirements/jetson.txt

# 运行
python main.py --profile edge_minimal --device cpu --backend onnx
```

### RK3588 / 其他 ARM Edge

```bash
pip install ncnn
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

# 输出示例:
# [PERF] fps=22.4 detect=18.2ms track=1.1ms recog_queue=2 skipped=38
# [FallWorker] frame=#300 detected=1 tracks=1
# [RECOG] enqueue=3 skip=42 reject=18 cache_hit=21 queue=1 worker=23.5ms
```

### Dependency Layering

```
requirements/
├── base.txt          # numpy, pyyaml, opencv-python-headless, scipy
├── edge_cpu.txt      # base + insightface + onnxruntime (CPU)
├── desktop.txt       # base + insightface + onnxruntime-gpu + boxmot + ultralytics + lap
├── jetson.txt        # base + insightface (推荐TensorRT)
└── dev.txt           # base + pytest + black + ruff
```

**Why edge_cpu does NOT install onnxruntime-gpu:**
- 树莓派/RK3588 无 NVIDIA GPU
- onnxruntime-gpu wheel 在 ARM 上无预编译版本
- CPU 推理在 320px + 15f interval 下可稳定运行

**Why edge_minimal disables behavior modules:**
- BehaviorEngine + EventSystem + AlertManager 每帧 ~100 次规则计算
- 对边缘 CPU 是显著开销
- EventSystem 和 AlertManager 可单独开启 (摔倒告警需要)

**Why NCNN for edge:**
- ARM NEON 手写汇编优化，Winograd 3×3 卷积 ~1.8x 加速
- 树莓派 5 上 YOLO 320px: NCNN ~35ms vs ONNX ~100ms
- 零 PyTorch 依赖，内存占用更低 (~200MB vs ~800MB)

---

## Changelog

### v10.1 — 项目清理 (2026-05-19)

- **移除** `tests/` — 35 个测试文件 (22 单元测试 + 5 集成测试 + 2 运行器 + 报告)
- **移除** `docs/` — 3 个文档 (INTEGRATION_GUIDE + fall_detection_integration + 设计计划)
- **移除** `.pytest_cache/` + `.superpowers/` — 缓存和会话状态
- **移除** `core/detection/` — 旧版检测目录 (已被 core/detectors/ 替代)
- **修复** FallDetectionTask.run() disabled 状态 crash (AttributeError on None._worker)
- **修复** test_behavior_system loitering_threshold 与预期不匹配
- **更新** README — 移除测试徽章/章节, 项目结构精简, 所有数字对齐

### v10 — 摔倒检测融合 (2026-05-18)

- **新增** `plugins/fall_engine/` — 摔倒检测引擎 (fall_logic.py + features.py + backends/ + detection.py)
- **新增** `plugins/fall_detection.py` — FallDetectionTask (VisionTask 插件, 异步 Worker 包装)
- **新增** `core/workers/fall_detection_worker.py` — 异步摔倒检测线程 (body-to-body IoU 追踪)
- **新增** `plugins/fall_engine/backends/ncnn_backend.py` — NCNN ARM NEON 推理后端
- **新增** `plugins/fall_engine/backends/factory.py` — create_backend() 工厂 (onnx/ncnn/ultralytics)
- **新增** `plugins/fall_engine/config.py` — 运行时配置注入 (替换原全局 config.py)
- **新增** `models/yolov8n-pose.onnx` — YOLOv8-Pose ONNX 模型 (12.9 MB)
- **新增** `docs/INTEGRATION_GUIDE.md` — 融合使用指南 (已移除)
- **新增** `docs/specs/` — 设计文档 + 实现计划 (已移除)
- **修改** `core/pipeline/pipeline.py` — _run_tasks() + _render() 摔倒骨架/人体框/状态
- **修改** `core/rendering/renderer.py` — draw_skeleton() + draw_body_bbox() + draw_fall_status()
- **修改** `core/alert_manager.py` — fall_detected / fall_potential / fall_recovered 告警分支
- **修改** `main.py` — build_tasks() 导入路径 (plugins.fall_detection)
- **修改** `configs/default.yaml` — tasks.fall_detection 完整配置节 (含 fall/features/tracking子节)
- **修改** `configs/edge_minimal.yaml` — 边缘摔倒配置 (NCNN, 320px, interval=15)
- **修改** `requirements/base.txt` — +scipy
- **修改** `requirements/desktop.txt` — +ultralytics +lap
- **修复** aspect_ratio 方向 (h/w 非 w/h) — 几何检测 Path 1 恢复正常
- **修复** ONNX 后处理坐标归一化
- **修复** 事件去重 (状态变化才发, 防刷屏)
- **修复** 跨线程 _fall_tracks 访问 (person_tid 写入 last_results)
- **修复** skeleton keypoint 边界检查 (< 0 替换 <= 0)
- **结果**: 人脸识别 + 摔倒检测融合运行, 共用 PersonManager track_id, 身份+摔倒联动告警

### v9.4 — 多摄像头架构 (2026-05-15)

- **新增** `core/camera_pipeline.py` — 单摄像头独立处理线程
- **新增** `core/multi_camera_manager.py` — 多 CameraPipeline 管理
- **新增** `core/scheduler/global_inference_scheduler.py` — 令牌桶限制同时 detect
- **新增** `configs/cameras.yaml` — 多摄像头 YAML 配置
- **修改** `main.py` — `--multi-camera` + `--camera 0 1` 多源支持

### v9.3 — 真正边缘化 (2026-05-14)

- **新增** `core/tracking/iou_tracker.py` — 轻量 IoU 跟踪器 (纯 Python + numpy, 零 boxmot)
- **结果**: `pip install -r requirements/edge_cpu.txt` → 运行成功, 209/209 测试通过

### v9.2 — 边缘部署配置 + 依赖分层 (2026-05-14)

- **新增** 3 个 profile 配置文件 + 5 层依赖文件
- **新增** `--benchmark` 参数

### v9.1 — VisionTask 插件接口 (2026-05-14)

- **新增** `core/interfaces.py` — VisionTask ABC + VisionEvent
- **新增** `plugins/fall_detection.py` — 摔倒检测空实现 (v10 升级为真实模块)

### v9.0 — 识别性能优化 (2026-05-14)

- FaceQualityFilter / EmbeddingCache / 向量化搜索 / 识别调度器 v2

### v8.0 — 减法重构 + 行为层 (2026-05-14)

- 9 个 build_* 函数 / runtime 开关 / 可选模块支持

### v7.x — SCRFD + ByteTrack + 行为层 (2026-05-13)

- SCRFD 探测器 / Motion Gate / Frame Scheduler / TrackMemory / PersonManager
- TrajectoryAnalyzer + BehaviorEngine / RegionManager + EventSystem + AlertManager

---

## Roadmap

```
v10 ✅  Fall Detection fusion — YOLOv8-Pose + 4-path logic + NCNN backend (current)
v9  ✅  Recognition optimization, FaceQualityFilter, embedding cache
v8  ✅  Behavior layer, triggers, hard dedup
v7  ✅  TrackMemory Hungarian + lock, PersonManager Re-ID cache
v6  ✅  SCRFD detector, 5-point landmarks, InsightFace detection
v5  ✅  Detection降频, PersonManager rendering, tracker optimization
v4  ✅  Async recognition, buffalo_s, smart scheduler
v9.1 ✅  VisionTask plugin interface
v9.3 ✅  LightweightIoUTracker, true edge (zero boxmot)
v9.4 ✅  Multi-camera architecture
v11 🔜  TensorRT acceleration (SCRFD + YOLO 2-3x speedup)
v12 🔜  Web dashboard (FastAPI + WebSocket)
v13 🔜  Multi-camera RTSP streaming
v14 🔜  OpenVINO backend
v15 🔜  Mobile push notifications
```

---

## FAQ

<details>
<summary><b>How do I enable/disable fall detection?</b></summary>

Edit `configs/default.yaml`:
```yaml
tasks:
  fall_detection:
    enabled: true    # ON
    enabled: false   # OFF
```
When disabled, zero overhead — FallDetectionWorker not created.
</details>

<details>
<summary><b>FPS is too low / video is choppy</b></summary>

Enable motion gate in config (default ON). Set `detector.input_size: 320` for faster detection. Set `tasks.fall_detection.interval: 15` to reduce fall inference frequency. Use `buffalo_s` model.
</details>

<details>
<summary><b>Fall detection not triggering when I fall?</b></summary>

The system needs 3.5s sustained fall posture to confirm. Quick "fake falls" (<2s) will only trigger Potential Fall, not confirmed FALL. Ensure `runtime.enable_event_system: true` and `runtime.enable_alert: true` are set.
</details>

<details>
<summary><b>What backends are available for fall detection?</b></summary>

Three backends via `create_backend()`:
- `onnx` — ONNX Runtime (CPU/CUDA, default)
- `ncnn` — Tencent NCNN (ARM NEON, Raspberry Pi optimal)
- `ultralytics` — Ultralytics YOLO (PyTorch, GPU)

Switch in config: `tasks.fall_detection.backend: "ncnn"`
</details>

<details>
<summary><b>How to run on Raspberry Pi?</b></summary>

```bash
pip install ncnn
pip install -r requirements/edge_cpu.txt
python main.py --profile edge_minimal
```
Profile uses: NCNN backend, 320px input, interval=15, max 10 persons.
</details>

<details>
<summary><b>Two people show the same name</b></summary>

Increase `recognition_threshold` to 0.75-0.8. Hard dedup ensures only one box per registered name. System auto-reverifies every 300 frames.
</details>

<details>
<summary><b>Identity switches when people cross paths</b></summary>

Hungarian lock prevents ID swaps during close interaction. Re-ID cache (10s TTL) preserves identity across lost/recovered tracks. Separation trigger (0.5s) forces re-recognition after overlap ends.
</details>

<details>
<summary><b>Changed model, faces not recognized</b></summary>

buffalo_l and buffalo_s embeddings are incompatible. Re-register all faces via `register_face.py`.
</details>

<details>
<summary><b>Camera won't open</b></summary>

Try `python main.py --camera 0` for built-in, or `--camera 1` for phone. Default is 1 (DroidCam). Windows: Settings → Privacy → Camera → Allow apps.
</details>

<details>
<summary><b>InsightFace compile error</b></summary>

Install VS C++ Build Tools first. Then run:
```bash
pip install insightface-0.7.3\insightface-0.7.3
```
</details>

<details>
<summary><b>How does fall+identity linkage work?</b></summary>

PersonManager is the sole tracking source — face and pose share the same track_id. When FALL is detected, the Worker matches the body bbox to PersonManager's tracks by center-distance. Known person = "Byron fell!", unknown = "Stranger fell!".
</details>

---

## License

LYUN License

---

<p align="center">
  <sub>Built for edge AI and real-time vision monitoring.</sub>
</p>
