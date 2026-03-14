"""奖励工厂与导出。"""

from __future__ import annotations

from config import Config
from rewards.base_reward import BaseReward, REWARD_REGISTRY

# 研究线1（信号质量）：signal_sparse / signal_euclidean_immediate / signal_bfs_immediate
from rewards import signal_sparse as _signal_sparse  # noqa: F401
from rewards import signal_euclidean_immediate as _signal_euclidean_immediate  # noqa: F401
from rewards import signal_dfs_immediate as _signal_dfs_immediate  # noqa: F401
from rewards import signal_bfs_immediate as _signal_bfs_immediate  # noqa: F401

# 研究线2（发放时机）：timing_immediate / timing_accumulated_delay / timing_fully_delayed
from rewards import timing_immediate as _timing_immediate  # noqa: F401
from rewards import timing_accumulated_delay as _timing_accumulated_delay  # noqa: F401
from rewards import timing_fully_delayed as _timing_fully_delayed  # noqa: F401


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
