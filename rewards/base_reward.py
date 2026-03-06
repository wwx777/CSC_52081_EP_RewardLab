"""奖励函数抽象基类与注册机制。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Set, Tuple, Type

import numpy as np


class BaseReward(ABC):
    """所有奖励类都应继承该基类。"""

    def reset(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        maze: np.ndarray,
    ) -> None:
        """每个 episode 开始时调用，默认无需额外状态。"""
        return None

    @abstractmethod
    def compute(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        reached_goal: bool,
        visited: Set[Tuple[int, int]],
        steps: int,
        maze: np.ndarray,
    ) -> float:
        """计算当前时间步奖励。"""
        raise NotImplementedError


REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {}


def register_reward(name: str) -> Callable[[Type[BaseReward]], Type[BaseReward]]:
    """注册奖励类，保持工厂逻辑简单。"""

    def _decorator(cls: Type[BaseReward]) -> Type[BaseReward]:
        key = name.strip().lower()
        if not key:
            raise ValueError("奖励名称不能为空")
        REWARD_REGISTRY[key] = cls
        return cls

    return _decorator
