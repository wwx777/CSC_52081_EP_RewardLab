"""画两张折线图：signal 和 timing 两条研究线的 return 均值±方差（跨 seed）。"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SIGNAL_REWARDS = [
    "signal_sparse",
    "signal_euclidean_immediate",
    "signal_dfs_immediate",
    "signal_bfs_immediate",
]
TIMING_REWARDS = [
    "timing_immediate",
    "timing_accumulated_delay",
    "timing_fully_delayed",
]

LABELS = {
    "signal_sparse":              "Sparse",
    "signal_euclidean_immediate": "Euclidean",
    "signal_dfs_immediate":       "DFS",
    "signal_bfs_immediate":       "BFS",
    "timing_immediate":           "Immediate",
    "timing_accumulated_delay":   "Accumulated Delay",
    "timing_fully_delayed":       "Fully Delayed",
}

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3",
]


def load_reward(reward_type: str):
    """返回 (timesteps, mean_over_seeds, std_over_seeds)。"""
    paths = sorted(glob.glob(os.path.join(LOG_DIR, reward_type, "seed_*", "evaluations.npz")))
    if not paths:
        return None, None, None

    seed_curves = []
    for path in paths:
        d = np.load(path)
        # results: (n_evals, n_episodes) → 每个 eval point 取 episode 均值
        seed_curves.append(d["results"].mean(axis=1))
    timesteps = np.load(paths[0])["timesteps"]

    seed_curves = np.stack(seed_curves, axis=0)          # (n_seeds, n_evals)
    mean = seed_curves.mean(axis=0)
    std  = seed_curves.std(axis=0)
    return timesteps, mean, std


def plot_group(reward_list: list, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for reward, color in zip(reward_list, COLORS):
        ts, mean, std = load_reward(reward)
        if ts is None:
            print(f"  [skip] {reward}: no data")
            continue
        label = LABELS[reward]
        ax.plot(ts, mean, color=color, linewidth=2, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_group(
        SIGNAL_REWARDS,
        title="Signal Quality: Effect of Distance Metric",
        filename="signal_return_curves.png",
    )
    plot_group(
        TIMING_REWARDS,
        title="Reward Timing: Effect of When Reward is Delivered",
        filename="timing_return_curves.png",
    )
