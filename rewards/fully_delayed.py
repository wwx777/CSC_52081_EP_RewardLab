"""完全延迟奖励：仅在回合结束时发放整个 episode 的累计奖励。"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("fully_delayed")
class FullyDelayedReward(BaseReward):
    """中间步骤不给奖励，终止时一次性发放累计的即时奖励。"""

    def __init__(self):
        self._prev_dist: Optional[int] = None
        self._accumulator: float = 0.0
        self._dist_cache: Dict[Tuple[int, int], int] = {}

    @staticmethod
    def _build_distance_map(goal_pos: Tuple[int, int], maze: np.ndarray) -> Dict[Tuple[int, int], int]:
        h, w = maze.shape
        dist: Dict[Tuple[int, int], int] = {}
        if maze[goal_pos[0], goal_pos[1]] != 0:
            return dist
        q: deque = deque()
        q.append(goal_pos)
        dist[goal_pos] = 0
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            x, y = q.popleft()
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 0 and (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        return dist

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._dist_cache = self._build_distance_map(goal_pos, maze)
        self._prev_dist = self._dist_cache.get(agent_pos)
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
        curr_dist = self._dist_cache.get(agent_pos)

        progress = 0.0
        if self._prev_dist is not None and curr_dist is not None:
            progress = self._prev_dist - curr_dist
        self._prev_dist = curr_dist

        immediate = progress - 0.01
        if reached_goal:
            immediate += 5.0

        self._accumulator += immediate

        if episode_end:
            out = self._accumulator
            self._accumulator = 0.0
            return float(out)

        return 0.0
