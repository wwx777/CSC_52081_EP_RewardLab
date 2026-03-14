"""DFS 路径距离即时奖励：用 DFS 生成树深度做 shaping（精确但非最优路径）。"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from rewards.base_reward import BaseReward, register_reward


@register_reward("signal_dfs_immediate")
class DFSImmediateReward(BaseReward):
    """
    研究线1（信号质量）的对照组。
    使用 DFS 树深度做 shaping：距离是迷宫内的真实路径，但并非最短路。
    与 signal_bfs_immediate 对比，说明距离度量的"质量"（最优 vs 次优）对学习的影响。
    参数：progress_multiplier=1.0，step_penalty=-0.01，goal_bonus=+20.0。
    """

    def __init__(self):
        self._prev_dist: Optional[int] = None
        self._dist_cache: Dict[Tuple[int, int], int] = {}

    @staticmethod
    def _build_dfs_distance_map(goal_pos: Tuple[int, int], maze: np.ndarray) -> Dict[Tuple[int, int], int]:
        """从 goal 出发做 DFS，每个可走格的距离为其在 DFS 树中的深度。"""
        h, w = maze.shape
        dist: Dict[Tuple[int, int], int] = {}
        if maze[goal_pos[0], goal_pos[1]] != 0:
            return dist

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        stack = [(goal_pos, 0)]

        while stack:
            pos, depth = stack.pop()
            if pos in dist:
                continue
            dist[pos] = depth
            x, y = pos
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 0 and (nx, ny) not in dist:
                    stack.append(((nx, ny), depth + 1))

        return dist

    def reset(self, agent_pos: Tuple[int, int], goal_pos: Tuple[int, int], maze: np.ndarray) -> None:
        self._dist_cache = self._build_dfs_distance_map(goal_pos, maze)
        self._prev_dist = self._dist_cache.get(agent_pos)

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

        reward = 1.0 * progress - 0.01
        if reached_goal:
            reward += 20.0
        return float(reward)
