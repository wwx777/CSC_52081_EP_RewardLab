"""分析图：成功率曲线（与 reward 曲线分开看）。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from analysis.utils import (
    SIGNAL_REWARDS, TIMING_REWARDS, DISPLAY_NAMES, COLORS, LINESTYLES,
    FIGURES_DIR, load_eval_data,
)


def plot_group(reward_list: list, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for reward_type in reward_list:
        data = load_eval_data(reward_type)
        if data is None:
            print(f"  [skip] {reward_type}: 无数据")
            continue

        t    = data["timesteps"]
        mean = data["mean_success"] * 100   # 转为百分比
        std  = data["std_success"]  * 100
        label = f"{DISPLAY_NAMES[reward_type]} (n={data['n_seeds']})"

        ax.plot(t, mean,
                label=label,
                color=COLORS[reward_type],
                linestyle=LINESTYLES[reward_type],
                linewidth=2)
        ax.fill_between(t, mean - std, mean + std,
                        color=COLORS[reward_type], alpha=0.15)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"保存: {output_path}")


if __name__ == "__main__":
    plot_group(
        SIGNAL_REWARDS,
        title="Exp1: Signal Quality — Success Rate vs Timesteps",
        output_path=FIGURES_DIR / "success_rate_signal.png",
    )
    plot_group(
        TIMING_REWARDS,
        title="Exp2: Temporal Structure — Success Rate vs Timesteps",
        output_path=FIGURES_DIR / "success_rate_timing.png",
    )
