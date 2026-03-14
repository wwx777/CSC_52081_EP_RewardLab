"""分析脚本共用常量和数据加载工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
CHECKPOINTS_DIR = ROOT / "checkpoints"
FIGURES_DIR = ROOT / "analysis" / "figures"

# 研究线1：信号质量
SIGNAL_REWARDS = [
    "signal_sparse",
    "signal_euclidean_immediate",
    "signal_dfs_immediate",
    "signal_bfs_immediate",
]

# 研究线2：发放时机
TIMING_REWARDS = [
    "timing_immediate",
    "timing_accumulated_delay",
    "timing_fully_delayed",
]

ALL_REWARDS = SIGNAL_REWARDS + TIMING_REWARDS

DISPLAY_NAMES = {
    "signal_sparse":               "Sparse",
    "signal_euclidean_immediate":  "Euclidean Immediate",
    "signal_dfs_immediate":        "DFS Immediate",
    "signal_bfs_immediate":        "BFS Immediate",
    "timing_immediate":            "Immediate",
    "timing_accumulated_delay":    "Accumulated Delay",
    "timing_fully_delayed":        "Fully Delayed",
}

COLORS = {
    "signal_sparse":               "#e74c3c",
    "signal_euclidean_immediate":  "#f39c12",
    "signal_dfs_immediate":        "#e67e22",
    "signal_bfs_immediate":        "#2ecc71",
    "timing_immediate":            "#2ecc71",
    "timing_accumulated_delay":    "#3498db",
    "timing_fully_delayed":        "#9b59b6",
}

LINESTYLES = {
    "signal_sparse":               "--",
    "signal_euclidean_immediate":  "-.",
    "signal_dfs_immediate":        ":",
    "signal_bfs_immediate":        "-",
    "timing_immediate":            "-",
    "timing_accumulated_delay":    "--",
    "timing_fully_delayed":        "-.",
}


def load_eval_data(reward_type: str, max_steps: int = 200) -> Optional[dict]:
    """加载某个 reward 类型所有 seed 的评估数据，返回聚合结果。"""
    reward_dir = LOGS_DIR / reward_type
    if not reward_dir.exists():
        return None

    seeds_data = []
    for seed_dir in sorted(reward_dir.glob("seed_*")):
        npz = seed_dir / "evaluations.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        seeds_data.append({
            "timesteps": data["timesteps"],
            "rewards":   data["results"],      # (n_eval, n_episodes)
            "lengths":   data["ep_lengths"],   # (n_eval, n_episodes)
        })

    if not seeds_data:
        return None

    # 对齐各 seed 的 timesteps
    common = seeds_data[0]["timesteps"]
    for d in seeds_data[1:]:
        common = np.intersect1d(common, d["timesteps"])
    if common.size == 0:
        return None

    mean_rewards, mean_lengths, success_rates = [], [], []
    for d in seeds_data:
        mask = np.isin(d["timesteps"], common)
        r = d["rewards"][mask]   # (n_common, n_ep)
        l = d["lengths"][mask]   # (n_common, n_ep)
        mean_rewards.append(r.mean(axis=1))
        mean_lengths.append(l.mean(axis=1))
        success_rates.append((l < max_steps).mean(axis=1))

    r_mat = np.vstack(mean_rewards)
    s_mat = np.vstack(success_rates)

    return {
        "timesteps":     common,
        "mean_reward":   r_mat.mean(0),
        "std_reward":    r_mat.std(0),
        "mean_success":  s_mat.mean(0),
        "std_success":   s_mat.std(0),
        "n_seeds":       len(seeds_data),
    }


def load_ev_data(reward_type: str) -> Optional[dict]:
    """加载某个 reward 类型所有 seed 的 explained variance 数据。"""
    reward_dir = LOGS_DIR / reward_type
    if not reward_dir.exists():
        return None

    seeds_data = []
    for seed_dir in sorted(reward_dir.glob("seed_*")):
        npz = seed_dir / "explained_variance.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        seeds_data.append({
            "timesteps":         data["timesteps"],
            "explained_variance": data["explained_variance"],
        })

    if not seeds_data:
        return None

    common = seeds_data[0]["timesteps"]
    for d in seeds_data[1:]:
        common = np.intersect1d(common, d["timesteps"])
    if common.size == 0:
        return None

    ev_list = []
    for d in seeds_data:
        mask = np.isin(d["timesteps"], common)
        ev_list.append(d["explained_variance"][mask])

    ev_mat = np.vstack(ev_list)
    return {
        "timesteps": common,
        "mean_ev":   ev_mat.mean(0),
        "std_ev":    ev_mat.std(0),
        "n_seeds":   len(seeds_data),
    }


def best_model_path(reward_type: str, seed: int = 42) -> Optional[Path]:
    """返回指定 reward 和 seed 的 best_model 路径（不存在则返回 None）。"""
    p = CHECKPOINTS_DIR / reward_type / f"seed_{seed}" / "best_model.zip"
    return p if p.exists() else None
