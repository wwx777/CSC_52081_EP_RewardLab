"""完全延迟奖励：仅在回合结束时给总回报。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("fully_delayed")
class FullyDelayedReward(BaseReward):
    """中间步骤不给奖励，终点一次性给奖励。"""

    def __init__(self):
        self._start_dist = 0.0

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._start_dist = self._dist(agent_pos, goal_pos)

    def compute(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        reached_goal: bool,
        visited: Set[Tuple[int, int]],
        steps: int,
        maze: np.ndarray,
    ) -> float:
        if not reached_goal:
            return 0.0

        final_dist = self._dist(agent_pos, goal_pos)
        total_progress = self._start_dist - final_dist
        terminal_bonus = 5.0
        return float(total_progress + terminal_bonus)
