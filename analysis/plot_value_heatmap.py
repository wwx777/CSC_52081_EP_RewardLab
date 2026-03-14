"""分析图：Value Function 热力图——对比 Euclidean vs BFS 的 V(s) 分布。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from stable_baselines3 import PPO

from config import Config
from envs.maze_env import MazeEnv
from analysis.utils import DISPLAY_NAMES, FIGURES_DIR, best_model_path

# 对比最有意思的两对：euclidean vs bfs（信号质量对比）
COMPARE_REWARDS = [
    "signal_sparse",
    "signal_euclidean_immediate",
    "signal_bfs_immediate",
]
FIXED_SEED = 2026


def compute_value_map(model: PPO, env: MazeEnv) -> np.ndarray:
    """计算迷宫中每个可走格的 V(s)。"""
    maze = env._maze
    goal_pos = env._goal_pos
    size = maze.shape[0]
    value_map = np.full((size, size), np.nan)

    for i in range(size):
        for j in range(size):
            if maze[i, j] != 0:
                continue
            env._agent_pos = (i, j)
            env._visited = {(i, j)}
            obs_hwc = env._render_obs()                          # (H, W, C)
            obs_chw = np.transpose(obs_hwc, (2, 0, 1))[None]    # (1, C, H, W)
            obs_tensor, _ = model.policy.obs_to_tensor(obs_chw)
            with torch.no_grad():
                value = model.policy.predict_values(obs_tensor).cpu().numpy().item()
            value_map[i, j] = value

    return value_map


if __name__ == "__main__":
    cfg = Config()
    cfg.reward_type = COMPARE_REWARDS[0]
    env = MazeEnv(cfg)
    env.reset(seed=FIXED_SEED)
    maze = env._maze.copy()
    goal_pos = env._goal_pos

    available = []
    value_maps = []

    for rt in COMPARE_REWARDS:
        mp = best_model_path(rt)
        if mp is None:
            print(f"  [skip] {rt}: 模型不存在")
            continue
        print(f"计算 {rt} 的 value map ...")
        cfg.reward_type = rt
        env2 = MazeEnv(cfg)
        env2.reset(seed=FIXED_SEED)
        model = PPO.load(str(mp))
        vm = compute_value_map(model, env2)
        available.append(rt)
        value_maps.append(vm)

    if not available:
        print("没有可用模型，请先运行 run_main.py")
        sys.exit(0)

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, rt, vm in zip(axes, available, value_maps):
        # 背景：墙为黑色
        bg = np.zeros((*maze.shape, 3))
        bg[maze == 1] = [0.15, 0.15, 0.15]

        ax.imshow(bg, interpolation="nearest")

        # Value function 热力图（仅可走格）
        masked_vm = np.ma.masked_where(np.isnan(vm), vm)
        vmin, vmax = np.nanmin(vm), np.nanmax(vm)
        im = ax.imshow(masked_vm, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       interpolation="nearest", alpha=0.85)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 标记起点和终点
        gx, gy = goal_pos
        ax.plot(gy, gx, "g*", markersize=14, label="Goal")
        ax.plot(1, 1, "b^", markersize=10, label="Start")

        ax.set_title(DISPLAY_NAMES[rt], fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Value Function Heatmap (maze seed={FIXED_SEED})", fontsize=13)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "value_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"保存: {out}")
