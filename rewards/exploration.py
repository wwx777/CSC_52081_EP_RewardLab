"""探索奖励实现。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward


class ExplorationReward(BaseReward):
    """鼓励访问新区域的奖励。"""

    def __init__(self):
        self._seen: Set[Tuple[int, int]] = set()

    def reset(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        maze: np.ndarray,
    ) -> None:
        self._seen = {agent_pos}

    def compute(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        reached_goal: bool,
        visited: Set[Tuple[int, int]],
        steps: int,
        maze: np.ndarray,
    ) -> float:
        reward = -0.005
        if agent_pos not in self._seen:
            reward += 0.1
            self._seen.add(agent_pos)
        if reached_goal:
            reward += 5.0
        return float(reward)
