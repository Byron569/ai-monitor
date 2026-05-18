# AI 智能监控系统 — 使用指南

## 快速启动

### 安装依赖

```bash
# 桌面 GPU 环境
pip install -r requirements/desktop.txt

# 边缘 CPU 环境 (树莓派 / RK3588)
pip install -r requirements/edge_cpu.txt
```

### 运行

```bash
# 仅人脸识别 (默认, 摔倒检测关闭)
python main.py

# 启用人脸识别 + 摔倒检测 (需先在 configs/default.yaml 中设置 enabled: true)
python main.py

# Benchmark 模式 (无渲染, 300 帧后自动退出)
python main.py --benchmark

# 指定摄像头
python main.py --camera 0

# 多摄像头
python main.py --camera 0 1 2
```

## 配置说明

编辑 `configs/default.yaml` 中的 `tasks.fall_detection`:

```yaml
tasks:
  fall_detection:
    enabled: true               # 开启摔倒检测
    backend: onnx               # onnx (边缘推荐) 或 ultralytics (GPU)
    device: cpu                 # cpu / cuda / auto
    interval: 5                 # 每 5 帧推理一次
```

## 模块接口

### FallDetectionTask

- 输入: frame (BGR), tracks (来自 PersonManager), context
- 输出: List[VisionEvent]
  - event_type: "fall_detected" / "fall_potential"
  - track_id: 与 Face ID 共用
  - confidence: 摔倒置信度
  - payload: { fall_state, bbox, keypoints }

### VisionEvent 事件类型

| event_type | 说明 |
|---|---|
| fall_detected | 确认摔倒 |
| fall_potential | 疑似摔倒 (预警) |
| fall_recovered | 摔倒后恢复 |

## 性能优化

- 边缘设备建议使用 ONNX 后端: `backend: onnx`
- 降低推理频率: `interval: 10`
- 减小输入尺寸: `input_size: 320`

## 多摄像头模式

```bash
python main.py --camera 0 1 2
```
