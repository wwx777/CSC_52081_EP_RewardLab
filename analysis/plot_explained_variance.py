"""分析图：Explained Variance 曲线——衡量 credit assignment 困难程度。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from analysis.utils import (
    TIMING_REWARDS, DISPLAY_NAMES, COLORS, LINESTYLES, FIGURES_DIR, load_ev_data,
)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(9, 5))
    found = False

    for rt in TIMING_REWARDS:
        data = load_ev_data(rt)
        if data is None:
            print(f"  [skip] {rt}: 无 explained_variance 数据")
            continue

        found = True
        t    = data["timesteps"]
        mean = data["mean_ev"]
        std  = data["std_ev"]
        label = f"{DISPLAY_NAMES[rt]} (n={data['n_seeds']})"

        ax.plot(t, mean,
                label=label,
                color=COLORS[rt],
                linestyle=LINESTYLES[rt],
                linewidth=2)
        ax.fill_between(t, mean - std, mean + std,
                        color=COLORS[rt], alpha=0.15)

    if not found:
        print("没有 explained_variance 数据，请先运行 run_main.py")
        sys.exit(0)

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Exp2: Explained Variance — Credit Assignment Quality", fontsize=13)
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Explained Variance")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "explained_variance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"保存: {out}")
