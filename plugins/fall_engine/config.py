"""fall_engine 配置模块 — FallDetectionTask 初始化时注入配置。"""

FALL = {}
FEATURES = {}
TRACKING = {}
CROSS_CAM = {}
CAM_PROC = {}


def set_config(cfg: dict):
    """由 FallDetectionTask.__init__() 调用，注入配置值。"""
    global FALL, FEATURES, TRACKING, CROSS_CAM, CAM_PROC
    FALL = cfg.get("fall", {})
    FEATURES = cfg.get("features", {})
    TRACKING = cfg.get("tracking", {})
    CROSS_CAM = cfg.get("cross_camera", {})
    CAM_PROC = cfg.get("camera_process", {})
