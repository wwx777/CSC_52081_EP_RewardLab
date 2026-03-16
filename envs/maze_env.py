"""迷宫环境定义。"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from envs.maze_generator import generate_dead_end_maze, generate_long_corridor_maze, generate_maze
from rewards import build_reward


class MazeEnv(gym.Env):
    """基于 RGB 观察的迷宫导航环境。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.reward_fn = build_reward(self.cfg)

        self.maze_size = self.cfg.maze_size if self.cfg.maze_size % 2 == 1 else self.cfg.maze_size + 1
        self.cell_size = self.cfg.cell_size
        self.max_steps = self.cfg.max_steps

        obs_h = self.maze_size * self.cell_size
        obs_w = self.maze_size * self.cell_size

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_h, obs_w, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng()
        self._maze: Optional[np.ndarray] = None
        self._agent_pos: Tuple[int, int] = (1, 1)
        self._goal_pos: Tuple[int, int] = (1, 1)
        self._visited: Set[Tuple[int, int]] = set()
        self._steps = 0

        self._moves = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
        }

    def _find_start_goal(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """分别从左上和右下找到最近可行走格。"""
        assert self._maze is not None
        size = self._maze.shape[0]

        # 找左上最近可行走格
        start = None
        for dist in range(2 * size):
            for i in range(size):
                j = dist - i
                if 0 <= j < size and self._maze[i, j] == 0:
                    start = (i, j)
                    break
            if start is not None:
                break

        # 找右下最近可行走格
        goal = None
        for dist in range(2 * size):
            for i in range(size - 1, -1, -1):
                j = (2 * size - 2 - dist) - i
                if 0 <= j < size and self._maze[i, j] == 0:
                    goal = (i, j)
                    break
            if goal is not None:
                break

        if start is None or goal is None:
            raise RuntimeError("迷宫中未找到合法的起点或终点")

        return start, goal

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境并生成新迷宫。"""
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        maze_seed = int(self._rng.integers(0, 2**31 - 1))
        variant = getattr(self.cfg, "maze_variant", "standard")
        if variant == "dead_end_dense":
            self._maze = generate_dead_end_maze(self.maze_size, seed=maze_seed)
        elif variant == "long_corridor":
            self._maze = generate_long_corridor_maze(self.maze_size, seed=maze_seed)
        else:
            self._maze = generate_maze(self.maze_size, seed=maze_seed)
        self._agent_pos, self._goal_pos = self._find_start_goal()

        self._steps = 0
        self._visited = {self._agent_pos}
        self.reward_fn.reset(self._agent_pos, self._goal_pos, self._maze)

        obs = self._render_obs()
        info = {"agent_pos": self._agent_pos, "goal_pos": self._goal_pos}
        return obs, info

    def step(self, action: int):
        """执行一步动作并计算奖励。"""
        assert self._maze is not None, "请先调用 reset()"

        action = int(action)
        dx, dy = self._moves.get(action, (0, 0))

        x, y = self._agent_pos
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size and self._maze[nx, ny] == 0:
            self._agent_pos = (nx, ny)

        self._steps += 1
        self._visited.add(self._agent_pos)

        reached_goal = self._agent_pos == self._goal_pos
        terminated = reached_goal
        truncated = self._steps >= self.max_steps

        reward = float(
            self.reward_fn.compute(
                agent_pos=self._agent_pos,
                goal_pos=self._goal_pos,
                reached_goal=reached_goal,
                episode_end=terminated or truncated,
                visited=self._visited,
                steps=self._steps,
                maze=self._maze,
            )
        )

        obs = self._render_obs()
        info = {
            "agent_pos": self._agent_pos,
            "goal_pos": self._goal_pos,
            "steps": self._steps,
            "reached_goal": reached_goal,
        }
        return obs, reward, terminated, truncated, info

    def _render_obs(self) -> np.ndarray:
        """渲染当前迷宫状态为 RGB 图像。"""
        assert self._maze is not None

        wall_color = np.array([40, 40, 40], dtype=np.uint8)
        path_color = np.array([240, 235, 220], dtype=np.uint8)
        agent_color = np.array([70, 130, 230], dtype=np.uint8)
        goal_color = np.array([60, 200, 100], dtype=np.uint8)
        visited_color = np.array([200, 215, 240], dtype=np.uint8)

        size = self._maze.shape[0]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        canvas[self._maze == 1] = wall_color
        canvas[self._maze == 0] = path_color

        for vx, vy in self._visited:
            if (vx, vy) != self._agent_pos and (vx, vy) != self._goal_pos:
                canvas[vx, vy] = visited_color

        gx, gy = self._goal_pos
        ax, ay = self._agent_pos
        canvas[gx, gy] = goal_color
        canvas[ax, ay] = agent_color

        # 按 cell_size 放大到像素级图像。
        obs = np.repeat(np.repeat(canvas, self.cell_size, axis=0), self.cell_size, axis=1)
        return obs

    def render(self):
        """返回当前帧。"""
        return self._render_obs()

    def close(self):
        """关闭环境资源。"""
        return None
