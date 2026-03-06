"""迷宫生成工具。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _ensure_odd_size(size: int) -> int:
    """确保迷宫尺寸为奇数且至少为 5。"""
    size = max(5, int(size))
    if size % 2 == 0:
        size += 1
    return size


def generate_maze(size: int, seed: int | None = None) -> np.ndarray:
    """使用递归回溯法生成迷宫。

    返回值中 1 表示墙，0 表示可通行区域。
    """
    rng = np.random.default_rng(seed)
    size = _ensure_odd_size(size)

    maze = np.ones((size, size), dtype=np.uint8)

    # 从 (1,1) 开始挖通道。
    start = (1, 1)
    maze[start] = 0

    stack: List[Tuple[int, int]] = [start]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        x, y = stack[-1]
        neighbors = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            nx, ny, dx, dy = neighbors[rng.integers(0, len(neighbors))]
            # 打通当前格与目标格中间的墙。
            maze[x + dx // 2, y + dy // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    return maze


def get_free_cells(maze: np.ndarray) -> List[Tuple[int, int]]:
    """返回迷宫中所有可行走坐标。"""
    coords = np.argwhere(maze == 0)
    return [tuple(map(int, c)) for c in coords]
