"""分析图：样本效率——各 reward 首次成功所需 timestep 柱状图。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from analysis.utils import (
    ALL_REWARDS, DISPLAY_NAMES, COLORS, FIGURES_DIR, load_eval_data,
)
from config import Config


def first_success_timestep(reward_type: str) -> float:
    """返回首次成功（success_rate > 0）对应的 timestep，未成功返回 inf。"""
    data = load_eval_data(reward_type, max_steps=Config().max_steps)
    if data is None:
        return float("inf")
    idx = np.where(data["mean_success"] > 0)[0]
    return float(data["timesteps"][idx[0]]) if len(idx) > 0 else float("inf")


if __name__ == "__main__":
    labels, values, colors = [], [], []
    max_ts = Config().total_timesteps

    for rt in ALL_REWARDS:
        ts = first_success_timestep(rt)
        labels.append(DISPLAY_NAMES[rt])
        values.append(ts if ts != float("inf") else max_ts)
        colors.append(COLORS[rt])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, [v / 1e6 for v in values], color=colors, edgecolor="white", linewidth=0.8)

    # 标注未收敛
    for bar, v in zip(bars, values):
        if v >= max_ts:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    "N/A", ha="center", va="bottom", fontsize=9, color="gray")

    ax.axvline(x=2.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "Exp1\n(Signal Quality)", ha="center", fontsize=9, alpha=0.6)
    ax.text(3.5, ax.get_ylim()[1] * 0.95, "Exp2\n(Timing)", ha="center", fontsize=9, alpha=0.6)

    ax.set_title("Sample Efficiency: First Successful Evaluation Timestep", fontsize=13)
    ax.set_ylabel("Timesteps (M)")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "sample_efficiency.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"保存: {out}")
