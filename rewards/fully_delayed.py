"""完全延迟奖励：仅在回合结束时发放整个 episode 的累计奖励。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("fully_delayed")
class FullyDelayedReward(BaseReward):
    """中间步骤不给奖励，终止时一次性发放累计的即时奖励。"""

    def __init__(self):
        self._prev_dist = 0.0
        self._accumulator = 0.0

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._prev_dist = self._dist(agent_pos, goal_pos)
        self._accumulator = 0.0

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
        immediate = (self._prev_dist - curr_dist) - 0.01
        self._prev_dist = curr_dist

        if reached_goal:
            immediate += 5.0

        self._accumulator += immediate

        if episode_end:
            out = self._accumulator
            self._accumulator = 0.0
            return float(out)

        return 0.0
