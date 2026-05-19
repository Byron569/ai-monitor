"""fall_engine 配置模块 — per-camera 配置隔离。"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class FallEngineConfig:
    fall: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    tracking: Dict[str, Any] = field(default_factory=dict)
    cross_cam: Dict[str, Any] = field(default_factory=dict)
    cam_proc: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            fall=cfg.get("fall", {}),
            features=cfg.get("features", {}),
            tracking=cfg.get("tracking", {}),
            cross_cam=cfg.get("cross_camera", {}),
            cam_proc=cfg.get("camera_process", {}),
        )


# Default instance (updated by FallDetectionTask at init)
_current: FallEngineConfig = FallEngineConfig()


def get_config() -> FallEngineConfig:
    return _current


def set_config(cfg: dict):
    global _current
    _current = FallEngineConfig.from_dict(cfg)
