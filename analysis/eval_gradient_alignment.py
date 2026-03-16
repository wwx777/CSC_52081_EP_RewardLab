"""Post-hoc evaluation: value gradient direction vs wall alignment.

对每个训练好的模型，在迷宫的每个可行走格子上：
1. 计算 4 个方向邻格的 V(s) 值，取 max-V 方向为 "policy gradient 方向"
2. 计算 wall misalignment rate：max-V 方向指向墙的比例
3. 计算 BFS alignment rate：max-V 方向与 BFS 最优方向一致的比例

无需重新训练，直接使用现有 checkpoint。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from collections import deque
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from analysis.utils import ALL_REWARDS, COLORS, DISPLAY_NAMES, FIGURES_DIR, best_model_path
from config import Config
from envs.maze_env import MazeEnv

SEEDS = [42, 43, 44]
N_MAZES = 20       # 每个 seed 采样多少个不同迷宫做分析
MAZE_EVAL_SEED_BASE = 9000   # 与训练/评估用的 seed 区分开

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right


def build_bfs_dist(maze: np.ndarray, goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """从 goal 出发 BFS，返回所有可达格子的距离字典。"""
    h, w = maze.shape
    dist: Dict[Tuple[int, int], int] = {goal: 0}
    q: deque = deque([goal])
    while q:
        x, y = q.popleft()
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 0 and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def get_value(model: PPO, obs_chw: np.ndarray) -> float:
    """从 CHW uint8 观测（值域 0-255）计算 V(s)。

    SB3 的 preprocess_obs 会在内部除以 255，所以这里传原始 float32 即可。
    """
    obs_batch = obs_chw[np.newaxis].astype(np.float32)   # (1, C, H, W)
    obs_tensor = torch.as_tensor(obs_batch, device=model.device)
    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)
    return float(value.item())


def analyze_one_maze(
    env: MazeEnv,
    model: PPO,
    maze_seed: int,
) -> Optional[Dict]:
    """在单张迷宫上统计 gradient alignment 指标。"""
    env.reset(seed=maze_seed)
    maze = env._maze.copy()
    goal = env._goal_pos

    bfs_dist = build_bfs_dist(maze, goal)
    h, w = maze.shape

    wall_misaligned = 0
    bfs_aligned = 0
    total = 0

    for r in range(h):
        for c in range(w):
            if maze[r, c] != 0 or (r, c) == goal:
                continue
            if (r, c) not in bfs_dist:
                continue

            # --- 为当前格子构造观测 ---
            env._agent_pos = (r, c)
            env._visited = {(r, c)}
            env._steps = 0

            # 计算 4 个方向邻格的 V 值
            neighbor_values: List[float] = []
            for dx, dy in MOVES:
                nr, nc = r + dx, c + dy
                if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == 0:
                    env._agent_pos = (nr, nc)
                else:
                    env._agent_pos = (r, c)   # 撞墙，位置不变

                env._visited = {env._agent_pos}
                obs_hwc = env._render_obs()
                obs_chw = np.transpose(obs_hwc, (2, 0, 1))
                neighbor_values.append(get_value(model, obs_chw))

            # 恢复 agent 到 (r, c)
            env._agent_pos = (r, c)

            best_dir = int(np.argmax(neighbor_values))
            br, bc = r + MOVES[best_dir][0], c + MOVES[best_dir][1]

            # wall misalignment: max-V 方向是否指向墙
            is_wall = not (0 <= br < h and 0 <= bc < w and maze[br, bc] == 0)

            # BFS optimal direction: 哪个方向的邻格 BFS 距离最小
            bfs_best_dir = None
            bfs_best_d = float("inf")
            for i, (dx, dy) in enumerate(MOVES):
                nr, nc = r + dx, c + dy
                if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == 0:
                    d = bfs_dist.get((nr, nc), float("inf"))
                    if d < bfs_best_d:
                        bfs_best_d = d
                        bfs_best_dir = i

            total += 1
            if is_wall:
                wall_misaligned += 1
            if bfs_best_dir is not None and best_dir == bfs_best_dir:
                bfs_aligned += 1

    if total == 0:
        return None

    return {
        "wall_misalignment_rate": wall_misaligned / total,
        "bfs_alignment_rate":     bfs_aligned / total,
        "n_cells":                total,
    }


def eval_one(reward_type: str, seed: int) -> Optional[Dict]:
    model_p = best_model_path(reward_type, seed)
    if model_p is None:
        print(f"  [skip] {reward_type} seed={seed}: 模型不存在")
        return None

    cfg = Config()
    cfg.reward_type = reward_type
    env = MazeEnv(cfg)
    model = PPO.load(str(model_p))

    wall_mis_rates: List[float] = []
    bfs_align_rates: List[float] = []

    for i in range(N_MAZES):
        maze_seed = MAZE_EVAL_SEED_BASE + seed * 1000 + i
        res = analyze_one_maze(env, model, maze_seed)
        if res:
            wall_mis_rates.append(res["wall_misalignment_rate"])
            bfs_align_rates.append(res["bfs_alignment_rate"])

    env.close()

    if not wall_mis_rates:
        return None

    return {
        "wall_misalignment_rate": float(np.mean(wall_mis_rates)),
        "bfs_alignment_rate":     float(np.mean(bfs_align_rates)),
    }


def main():
    print(f"=== Value Gradient Alignment Evaluation ({N_MAZES} mazes/seed) ===\n")
    results: Dict[str, Dict] = {}

    for rt in ALL_REWARDS:
        print(f"[{rt}]")
        seed_results = []
        for seed in SEEDS:
            r = eval_one(rt, seed)
            if r:
                seed_results.append(r)
                print(
                    f"  seed={seed}: wall_misalign={r['wall_misalignment_rate']:.3f}"
                    f"  bfs_align={r['bfs_alignment_rate']:.3f}"
                )

        if not seed_results:
            print(f"  -> 跳过\n")
            continue

        results[rt] = {
            "wall_misalignment_rate": float(np.mean([s["wall_misalignment_rate"] for s in seed_results])),
            "wall_mis_std":           float(np.std([s["wall_misalignment_rate"] for s in seed_results])),
            "bfs_alignment_rate":     float(np.mean([s["bfs_alignment_rate"] for s in seed_results])),
            "bfs_align_std":          float(np.std([s["bfs_alignment_rate"] for s in seed_results])),
        }
        print(
            f"  -> 汇总: wall_mis={results[rt]['wall_misalignment_rate']:.3f}±{results[rt]['wall_mis_std']:.3f}"
            f"  bfs_align={results[rt]['bfs_alignment_rate']:.3f}±{results[rt]['bfs_align_std']:.3f}\n"
        )

    if not results:
        print("没有可用模型，请先运行训练。")
        return

    # ---- 绘图 ----
    reward_types = list(results.keys())
    labels = [DISPLAY_NAMES.get(rt, rt) for rt in reward_types]
    colors = [COLORS.get(rt, "#888888") for rt in reward_types]
    x = np.arange(len(reward_types))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Value Gradient Alignment Analysis (Deterministic Policy)", fontsize=13)

    # Wall misalignment rate
    ax = axes[0]
    means = [results[rt]["wall_misalignment_rate"] * 100 for rt in reward_types]
    stds  = [results[rt]["wall_mis_std"] * 100         for rt in reward_types]
    ax.bar(x, means, color=colors, edgecolor="white", yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Wall Misalignment Rate (%)")
    ax.set_title("Gradient → Wall (lower = better)")
    ax.grid(axis="y", alpha=0.3)

    # BFS alignment rate
    ax = axes[1]
    means = [results[rt]["bfs_alignment_rate"] * 100 for rt in reward_types]
    stds  = [results[rt]["bfs_align_std"] * 100       for rt in reward_types]
    ax.bar(x, means, color=colors, edgecolor="white", yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("BFS Alignment Rate (%)")
    ax.set_title("Gradient ↔ BFS Optimal (higher = better)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "gradient_alignment.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"图表已保存: {out}")

    npz_out = FIGURES_DIR / "gradient_alignment.npz"
    np.savez(
        npz_out,
        reward_types=np.array(reward_types),
        wall_misalignment_rate=np.array([results[rt]["wall_misalignment_rate"] for rt in reward_types]),
        bfs_alignment_rate=np.array([results[rt]["bfs_alignment_rate"] for rt in reward_types]),
    )
    print(f"数值结果已保存: {npz_out}")


if __name__ == "__main__":
    main()
