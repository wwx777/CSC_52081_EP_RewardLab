"""Post-hoc evaluation: path optimality ratio = actual_steps / bfs_shortest.

用现有 checkpoint 计算每个 reward 类型的路径最优性比率。
比率越接近 1 表示 agent 走的路越接近最短路径；越大表示越绕远路。
无需重新训练。
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
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from analysis.utils import ALL_REWARDS, COLORS, DISPLAY_NAMES, FIGURES_DIR, best_model_path
from config import Config
from envs.maze_env import MazeEnv

SEEDS = [42, 43, 44]
N_EPISODES = 50


def bfs_shortest(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
    """BFS 计算 start→goal 最短路径步数，不可达返回 None。"""
    if start == goal:
        return 0
    h, w = maze.shape
    q: deque = deque([(start, 0)])
    visited = {start}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        (x, y), dist = q.popleft()
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 0 and (nx, ny) not in visited:
                if (nx, ny) == goal:
                    return dist + 1
                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))
    return None


def eval_one(reward_type: str, seed: int, n_episodes: int = N_EPISODES) -> Optional[Dict]:
    model_p = best_model_path(reward_type, seed)
    if model_p is None:
        print(f"  [skip] {reward_type} seed={seed}: 模型不存在")
        return None

    cfg = Config()
    cfg.reward_type = reward_type

    vec_env = make_vec_env(MazeEnv, n_envs=1, env_kwargs={"cfg": cfg}, seed=seed * 1000)
    vec_env = VecTransposeImage(vec_env)
    model = PPO.load(str(model_p))

    ratios: List[float] = []
    success_ratios: List[float] = []  # 仅成功回合的比率
    all_ratios: List[float] = []      # 所有回合（失败=max_steps/bfs）

    for _ in range(n_episodes):
        obs = vec_env.reset()

        # 从内部环境读取迷宫结构（make_vec_env 会套 Monitor，需要 .env 才到 MazeEnv）
        _wrapped = vec_env.venv.envs[0]  # type: ignore[attr-defined]
        inner: MazeEnv = _wrapped.env if hasattr(_wrapped, "env") else _wrapped
        maze = inner._maze.copy()
        start = inner._agent_pos
        goal = inner._goal_pos

        bfs_len = bfs_shortest(maze, start, goal)
        if bfs_len is None or bfs_len == 0:
            continue

        done = False
        actual_steps = 0
        reached = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            actual_steps += 1
            if done:
                reached = bool(infos[0].get("reached_goal", False))

        ratio = actual_steps / bfs_len
        all_ratios.append(ratio)
        if reached:
            success_ratios.append(ratio)

    vec_env.close()

    if not all_ratios:
        return None

    return {
        "mean_ratio":         float(np.mean(all_ratios)),
        "median_ratio":       float(np.median(all_ratios)),
        "mean_success_ratio": float(np.mean(success_ratios)) if success_ratios else float("nan"),
        "success_rate":       len(success_ratios) / len(all_ratios),
        "n_valid":            len(all_ratios),
    }


def main():
    print("=== Path Optimality Ratio Evaluation ===\n")
    results: Dict[str, Dict] = {}

    for rt in ALL_REWARDS:
        print(f"[{rt}]")
        seed_results = []
        for seed in SEEDS:
            r = eval_one(rt, seed)
            if r:
                seed_results.append(r)
                print(f"  seed={seed}: ratio={r['mean_ratio']:.3f}  success_ratio={r['mean_success_ratio']:.3f}  success={r['success_rate']:.2%}")

        if not seed_results:
            print(f"  -> 跳过（无可用模型）\n")
            continue

        results[rt] = {
            "mean_ratio":         float(np.mean([s["mean_ratio"] for s in seed_results])),
            "std_ratio":          float(np.std([s["mean_ratio"] for s in seed_results])),
            "mean_success_ratio": float(np.nanmean([s["mean_success_ratio"] for s in seed_results])),
            "std_success_ratio":  float(np.nanstd([s["mean_success_ratio"] for s in seed_results])),
        }
        print(f"  -> 汇总: ratio={results[rt]['mean_ratio']:.3f}±{results[rt]['std_ratio']:.3f}\n")

    if not results:
        print("没有可用模型，请先运行训练。")
        return

    # ---- 绘图 ----
    reward_types = list(results.keys())
    labels = [DISPLAY_NAMES.get(rt, rt) for rt in reward_types]
    colors = [COLORS.get(rt, "#888888") for rt in reward_types]
    x = np.arange(len(reward_types))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Path Optimality Ratio  (actual steps / BFS shortest path)", fontsize=13)

    # 全部回合的 ratio
    ax = axes[0]
    means = [results[rt]["mean_ratio"] for rt in reward_types]
    stds  = [results[rt]["std_ratio"]  for rt in reward_types]
    bars = ax.bar(x, means, color=colors, edgecolor="white", yerr=stds, capsize=4)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="optimal (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Optimality Ratio (lower = better)")
    ax.set_title("All Episodes")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 仅成功回合的 ratio
    ax = axes[1]
    means_s = [results[rt]["mean_success_ratio"] for rt in reward_types]
    stds_s  = [results[rt]["std_success_ratio"]  for rt in reward_types]
    ax.bar(x, means_s, color=colors, edgecolor="white", yerr=stds_s, capsize=4)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="optimal (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Optimality Ratio (lower = better)")
    ax.set_title("Successful Episodes Only")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "path_optimality.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"图表已保存: {out}")

    # 保存数值结果
    npz_out = FIGURES_DIR / "path_optimality.npz"
    np.savez(
        npz_out,
        reward_types=np.array(reward_types),
        mean_ratio=np.array([results[rt]["mean_ratio"] for rt in reward_types]),
        std_ratio=np.array([results[rt]["std_ratio"] for rt in reward_types]),
        mean_success_ratio=np.array([results[rt]["mean_success_ratio"] for rt in reward_types]),
    )
    print(f"数值结果已保存: {npz_out}")


if __name__ == "__main__":
    main()
