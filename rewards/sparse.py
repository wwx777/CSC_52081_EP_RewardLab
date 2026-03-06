"""稀疏奖励实现。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("sparse")
class SparseReward(BaseReward):
    """只有终点奖励，其他时间只给小惩罚。"""

    def compute(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        reached_goal: bool,
        visited: Set[Tuple[int, int]],
        steps: int,
        maze: np.ndarray,
    ) -> float:
        if reached_goal:
            return 1.0
        return -0.01
