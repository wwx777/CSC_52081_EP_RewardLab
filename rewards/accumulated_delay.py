"""累计延迟奖励：按固定间隔发放累积值。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("accumulated_delay")
class AccumulatedDelayReward(BaseReward):
    """将即时奖励累计后延迟发放，降低反馈频率。"""

    def __init__(self, delay_steps: int = 10):
        self.delay_steps = max(1, int(delay_steps))
        self._prev_dist = 0.0
        self._bucket = 0.0

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._prev_dist = self._dist(agent_pos, goal_pos)
        self._bucket = 0.0

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

        self._bucket += immediate

        if episode_end or steps % self.delay_steps == 0:
            out = self._bucket
            self._bucket = 0.0
            return float(out)

        return 0.0
