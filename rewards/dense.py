"""稠密奖励实现（potential-based shaping）。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward


class DenseReward(BaseReward):
    """基于势函数差分的稠密奖励。"""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self._prev_potential = 0.0

    @staticmethod
    def _potential(agent_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> float:
        # Phi(s) = -||s-goal||_2
        return -float(np.linalg.norm(np.array(agent_pos, dtype=np.float32) - np.array(goal_pos, dtype=np.float32)))

    def reset(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        maze: np.ndarray,
    ) -> None:
        self._prev_potential = self._potential(agent_pos, goal_pos)

    def compute(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        reached_goal: bool,
        visited: Set[Tuple[int, int]],
        steps: int,
        maze: np.ndarray,
    ) -> float:
        curr_potential = self._potential(agent_pos, goal_pos)
        shaping = self.gamma * curr_potential - self._prev_potential
        self._prev_potential = curr_potential

        reward = shaping - 0.01
        if reached_goal:
            reward += 10.0
        return float(reward)
