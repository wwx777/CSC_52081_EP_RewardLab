"""即时奖励：每一步都给反馈。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("immediate")
class ImmediateReward(BaseReward):
    """根据与目标距离的改善即时给奖励。"""

    def __init__(self):
        self._prev_dist = 0.0

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._prev_dist = self._dist(agent_pos, goal_pos)

    def compute(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        reached_goal: bool,
        episode_end: bool,
        visited: Set[Tuple[int, int]],
        steps: int,
        maze: np.ndarray,
    ) -> float:
        curr_dist = self._dist(agent_pos, goal_pos)
        progress = self._prev_dist - curr_dist
        self._prev_dist = curr_dist

        reward = progress - 0.01
        if reached_goal:
            reward += 5.0
        return float(reward)
