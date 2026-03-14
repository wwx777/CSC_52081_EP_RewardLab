"""欧氏距离即时奖励：用直线距离做 shaping（存在绕路陷阱）。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("euclidean_immediate")
class EuclideanImmediateReward(BaseReward):
    """
    研究线1（信号质量）的对照组。
    使用欧氏距离做 shaping，在迷宫中会因墙壁产生误导性梯度。
    参数与 bfs_immediate 对齐：progress_multiplier=1.0，step_penalty=-0.01，goal_bonus=+20.0。
    """

    def __init__(self):
        self._prev_dist: float = 0.0

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._prev_dist = self._euclidean(agent_pos, goal_pos)

    @staticmethod
    def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

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
        curr_dist = self._euclidean(agent_pos, goal_pos)
        progress = self._prev_dist - curr_dist
        self._prev_dist = curr_dist

        reward = 1.0 * progress - 0.01
        if reached_goal:
            reward += 20.0
        return float(reward)
