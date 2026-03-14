"""稀疏奖励：仅在终点给奖励，无过程引导。"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("sparse")
class SparseReward(BaseReward):
    """
    研究线1（信号质量）的下界基线。
    无任何距离引导，只有终点奖励。
    参数与其他 reward 对齐：step_penalty=-0.01，goal_bonus=+20.0。
    """

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
        if reached_goal:
            return 20.0
        return -0.01
