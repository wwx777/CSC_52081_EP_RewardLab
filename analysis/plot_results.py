from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
FIGURES_DIR = ROOT / "analysis" / "figures"
OUTPUT_PATH = FIGURES_DIR / "reward_comparison.png"


def load_seed_curve(npz_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    data = np.load(npz_path)
    timesteps = data.get("timesteps")
    results = data.get("results")
    if timesteps is None or results is None or len(timesteps) == 0:
        return None

    rewards = np.mean(results, axis=1)
    return np.asarray(timesteps), np.asarray(rewards)


def collect_reward_curves(logs_dir: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    if not logs_dir.exists():
        return curves

    for reward_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        reward_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for seed_dir in sorted(p for p in reward_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            npz_path = seed_dir / "evaluations.npz"
            if not npz_path.exists():
                continue

            curve = load_seed_curve(npz_path)
            if curve is not None:
                reward_curves.append(curve)

        if reward_curves:
            curves[reward_dir.name] = reward_curves

    return curves


def aggregate_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_steps = curves[0][0]
    for steps, _ in curves[1:]:
        common_steps = np.intersect1d(common_steps, steps)

    if common_steps.size == 0:
        raise ValueError("No common evaluation timesteps across seeds.")

    aligned_rewards = []
    for steps, rewards in curves:
        mask = np.isin(steps, common_steps)
        aligned_rewards.append(rewards[mask])

    reward_matrix = np.vstack(aligned_rewards)
    mean_rewards = reward_matrix.mean(axis=0)
    std_rewards = reward_matrix.std(axis=0)
    return common_steps, mean_rewards, std_rewards


def plot_reward_comparison(curves_by_reward: dict[str, list[tuple[np.ndarray, np.ndarray]]], output_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for reward_type, curves in curves_by_reward.items():
        steps, mean_rewards, std_rewards = aggregate_curves(curves)
        ax.plot(steps, mean_rewards, label=f"{reward_type} (n={len(curves)})")
        ax.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    ax.set_title("Reward Comparison")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Evaluation Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    curves_by_reward = collect_reward_curves(LOGS_DIR)
    if not curves_by_reward:
        print(f"No experiment logs found under {LOGS_DIR}")
        return

    plot_reward_comparison(curves_by_reward, OUTPUT_PATH)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
