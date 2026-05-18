"""
配置加载工具 — 读取 YAML 配置文件，提供类型安全的配置访问。
"""

import os
import yaml
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径。为 None 时自动查找 configs/default.yaml。

    Returns:
        dict: 配置字典。

    使用示例：
        config = load_config()
        camera_index = config["camera"]["index"]
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "default.yaml",
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> str:
    """返回项目根目录的绝对路径。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
