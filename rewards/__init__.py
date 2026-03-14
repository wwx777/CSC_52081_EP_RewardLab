"""奖励工厂与导出。"""

from __future__ import annotations

from config import Config
from rewards.base_reward import BaseReward, REWARD_REGISTRY

# 导入模块以触发注册。
from rewards import accumulated_delay as _accumulated_delay  # noqa: F401
from rewards import fully_delayed as _fully_delayed  # noqa: F401
from rewards import immediate as _immediate  # noqa: F401
from rewards import sparse as _sparse  # noqa: F401
from rewards import intermediate_delayed as _intermediate_delayed  # noqa: F401


def build_reward(cfg: Config) -> BaseReward:
    """根据配置构建奖励实例。"""
    raw = cfg.reward_type.lower().strip()
    reward_type = raw.replace("-", " ").replace("_", " ").strip()
    if reward_type.endswith(" reward"):
        reward_type = reward_type[: -len(" reward")].strip()
    reward_type = reward_type.replace(" ", "_")
    if reward_type not in REWARD_REGISTRY:
        valid = ", ".join(sorted(REWARD_REGISTRY.keys()))
        raise ValueError(f"未知 reward_type: {cfg.reward_type}，可选值: {valid}")
    return REWARD_REGISTRY[reward_type]()


__all__ = ["BaseReward", "build_reward", "REWARD_REGISTRY"]
