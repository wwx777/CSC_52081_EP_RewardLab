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


def generate_dead_end_maze(size: int, seed: int | None = None) -> np.ndarray:
    """使用随机 Prim 算法生成迷宫。

    Prim 算法从大量候选格中随机挑选，产生大量短分支，即 dead-end 密集型迷宫。
    返回值中 1 表示墙，0 表示可通行区域。
    """
    rng = np.random.default_rng(seed)
    size = _ensure_odd_size(size)

    maze = np.ones((size, size), dtype=np.uint8)
    start = (1, 1)
    maze[start] = 0

    visited: set = {start}
    frontier: List[Tuple[int, int]] = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    for dx, dy in directions:
        nx, ny = 1 + dx, 1 + dy
        if 1 <= nx < size - 1 and 1 <= ny < size - 1:
            frontier.append((nx, ny))

    while frontier:
        # 随机挑选 frontier 中的格子
        idx = int(rng.integers(0, len(frontier)))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        cx, cy = frontier.pop()

        if maze[cx, cy] == 0:
            continue

        # 找到已访问的邻格（步长为 2）
        neighbors = [
            (cx + dx, cy + dy)
            for dx, dy in directions
            if (cx + dx, cy + dy) in visited
        ]
        if not neighbors:
            continue

        # 随机选一个已访问邻格打通墙
        nx, ny = neighbors[int(rng.integers(0, len(neighbors)))]
        maze[cx, cy] = 0
        maze[(cx + nx) // 2, (cy + ny) // 2] = 0
        visited.add((cx, cy))

        # 把新格子的未访问邻格加入 frontier
        for dx, dy in directions:
            fx, fy = cx + dx, cy + dy
            if 1 <= fx < size - 1 and 1 <= fy < size - 1 and maze[fx, fy] == 1:
                frontier.append((fx, fy))

    return maze


def generate_long_corridor_maze(size: int, seed: int | None = None) -> np.ndarray:
    """使用带方向偏置的递归回溯法生成迷宫。

    给当前移动方向赋予更高权重，使 agent 倾向于直线延伸，产生长走廊型迷宫。
    返回值中 1 表示墙，0 表示可通行区域。
    """
    rng = np.random.default_rng(seed)
    size = _ensure_odd_size(size)

    maze = np.ones((size, size), dtype=np.uint8)
    start = (1, 1)
    maze[start] = 0

    stack: List[Tuple[int, int]] = [start]
    last_dir: Tuple[int, int] | None = None
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    SAME_DIR_WEIGHT = 5.0  # 沿当前方向继续的概率权重

    while stack:
        x, y = stack[-1]
        neighbors = [
            (x + dx, y + dy, dx, dy)
            for dx, dy in directions
            if 1 <= x + dx < size - 1 and 1 <= y + dy < size - 1 and maze[x + dx, y + dy] == 1
        ]

        if neighbors:
            weights = np.array([
                SAME_DIR_WEIGHT if last_dir is not None and (dx, dy) == last_dir else 1.0
                for _, _, dx, dy in neighbors
            ])
            weights /= weights.sum()
            idx = int(rng.choice(len(neighbors), p=weights))
            nx, ny, dx, dy = neighbors[idx]
            maze[x + dx // 2, y + dy // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
            last_dir = (dx, dy)
        else:
            stack.pop()
            last_dir = None

    return maze


def get_free_cells(maze: np.ndarray) -> List[Tuple[int, int]]:
    """返回迷宫中所有可行走坐标。"""
    coords = np.argwhere(maze == 0)
    return [tuple(map(int, c)) for c in coords]
