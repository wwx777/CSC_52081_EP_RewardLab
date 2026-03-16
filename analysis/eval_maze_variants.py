"""迷宫变体实验评估：Wall Hit Rate & Dead-end Dwell Time。

从 checkpoints_variants/ 加载模型，在对应迷宫变体上跑 deterministic rollout，
计算两个重点指标：
  1. Wall Hit Rate          — 撞墙步数占比（越低越好）
  2. Dead-end Dwell Time    — 每次进入 dead-end 平均停留步数（越低越好）

结果保存为图表和 .npz 数值文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from config import Config
from envs.maze_env import MazeEnv

# ---- 实验矩阵（与 run_maze_variants.py 保持一致） ----
MAZE_VARIANTS = ["dead_end_dense", "long_corridor"]
SIGNALS       = ["signal_euclidean_immediate", "signal_bfs_immediate"]
SEEDS         = [42, 43, 44]
N_EPISODES    = 50

SIGNAL_LABELS = {
    "signal_euclidean_immediate": "Euclidean",
    "signal_bfs_immediate":       "BFS",
}
VARIANT_LABELS = {
    "dead_end_dense": "Dead-end Dense",
    "long_corridor":  "Long Corridor",
}
SIGNAL_COLORS = {
    "signal_euclidean_immediate": "#f39c12",
    "signal_bfs_immediate":       "#2ecc71",
}

ROOT         = Path(__file__).resolve().parents[1]
CKPT_DIR     = ROOT / "checkpoints_variants"
FIGURES_DIR  = ROOT / "analysis" / "figures"


def make_run_name(maze_variant: str, signal: str) -> str:
    short = {"signal_euclidean_immediate": "euclidean", "signal_bfs_immediate": "bfs"}
    return f"variant_{maze_variant}_{short[signal]}"


def best_variant_model(run_name: str, seed: int) -> Optional[Path]:
    base = CKPT_DIR / run_name / f"seed_{seed}"
    for p in [base / f"ppo_{run_name}_final.zip", base / "best_model.zip"]:
        if p.exists():
            return p
    return None


# ---- Dead-end 检测 ----

def find_dead_ends(maze: np.ndarray) -> Set[Tuple[int, int]]:
    """返回所有 dead-end 格子（恰好 1 个开放邻格的非墙格子）。"""
    dead_ends: Set[Tuple[int, int]] = set()
    h, w = maze.shape
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if maze[r, c] == 0:
                open_nb = sum(
                    1 for dr, dc in moves
                    if 0 <= r + dr < h and 0 <= c + dc < w and maze[r + dr, c + dc] == 0
                )
                if open_nb == 1:
                    dead_ends.add((r, c))
    return dead_ends


# ---- 单次 seed 评估 ----

def eval_one(maze_variant: str, signal: str, seed: int, n_episodes: int = N_EPISODES) -> Optional[Dict]:
    run_name = make_run_name(maze_variant, signal)
    model_p  = best_variant_model(run_name, seed)
    if model_p is None:
        print(f"  [skip] {run_name} seed={seed}: 模型不存在")
        return None

    cfg = Config()
    cfg.reward_type  = signal
    cfg.maze_variant = maze_variant

    vec_env = make_vec_env(MazeEnv, n_envs=1, env_kwargs={"cfg": cfg}, seed=seed * 1000)
    vec_env = VecTransposeImage(vec_env)
    model   = PPO.load(str(model_p))

    wall_hit_rates:    List[float] = []
    dwell_times:       List[float] = []   # 每次进入 dead-end 的平均停留步数

    for _ in range(n_episodes):
        obs       = vec_env.reset()
        _wrapped = vec_env.venv.envs[0]  # type: ignore[attr-defined]
        inner: MazeEnv = _wrapped.env if hasattr(_wrapped, "env") else _wrapped
        maze      = inner._maze.copy()
        dead_ends = find_dead_ends(maze)

        done       = False
        prev_pos   = None
        wall_hits  = 0
        ep_steps   = 0

        in_dead_end        = False
        dead_end_entries   = 0
        dead_end_steps     = 0
        current_dwell      = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            done      = bool(dones[0])
            curr_pos  = infos[0].get("agent_pos")
            ep_steps += 1

            # Wall hit
            if prev_pos is not None and curr_pos == prev_pos:
                wall_hits += 1
            prev_pos = curr_pos

            # Dead-end dwell
            if curr_pos in dead_ends:
                dead_end_steps += 1
                current_dwell  += 1
                if not in_dead_end:
                    dead_end_entries += 1
                    in_dead_end = True
            else:
                if in_dead_end:
                    in_dead_end   = False
                    current_dwell = 0

        wall_hit_rates.append(wall_hits / ep_steps if ep_steps > 0 else 0.0)

        if dead_end_entries > 0:
            dwell_times.append(dead_end_steps / dead_end_entries)

    vec_env.close()

    return {
        "wall_hit_rate":      float(np.mean(wall_hit_rates)),
        "wall_hit_std":       float(np.std(wall_hit_rates)),
        "dead_end_dwell":     float(np.mean(dwell_times))     if dwell_times else float("nan"),
        "dead_end_dwell_std": float(np.std(dwell_times))      if dwell_times else float("nan"),
    }


# ---- 汇总 ----

def collect_results() -> Dict:
    """返回 results[maze_variant][signal] = 跨 seeds 汇总指标。"""
    results: Dict = {}

    for mv in MAZE_VARIANTS:
        results[mv] = {}
        for sig in SIGNALS:
            run_name = make_run_name(mv, sig)
            print(f"[{run_name}]")
            seed_data = []

            for seed in SEEDS:
                r = eval_one(mv, sig, seed)
                if r:
                    seed_data.append(r)
                    print(
                        f"  seed={seed}  wall={r['wall_hit_rate']:.3f}"
                        f"  dwell={r['dead_end_dwell']:.2f}"
                    )

            if not seed_data:
                print("  -> 跳过\n")
                results[mv][sig] = None
                continue

            results[mv][sig] = {
                "wall_hit_rate":      float(np.mean([s["wall_hit_rate"]  for s in seed_data])),
                "wall_hit_std":       float(np.std([s["wall_hit_rate"]   for s in seed_data])),
                "dead_end_dwell":     float(np.nanmean([s["dead_end_dwell"] for s in seed_data])),
                "dead_end_dwell_std": float(np.nanstd([s["dead_end_dwell"]  for s in seed_data])),
            }
            d = results[mv][sig]
            print(
                f"  -> wall={d['wall_hit_rate']:.3f}±{d['wall_hit_std']:.3f}"
                f"  dwell={d['dead_end_dwell']:.2f}±{d['dead_end_dwell_std']:.2f}\n"
            )

    return results


# ---- 绘图 ----

def plot_results(results: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Maze Variant Experiments: Wall Hit Rate & Dead-end Dwell Time", fontsize=13)

    for row, mv in enumerate(MAZE_VARIANTS):
        sig_labels = [SIGNAL_LABELS[s] for s in SIGNALS]
        colors     = [SIGNAL_COLORS[s] for s in SIGNALS]
        x          = np.arange(len(SIGNALS))

        # Wall Hit Rate
        ax = axes[row][0]
        means = [results[mv][sig]["wall_hit_rate"] * 100 if results[mv][sig] else 0 for sig in SIGNALS]
        stds  = [results[mv][sig]["wall_hit_std"]  * 100 if results[mv][sig] else 0 for sig in SIGNALS]
        ax.bar(x, means, color=colors, edgecolor="white", yerr=stds, capsize=5, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sig_labels, fontsize=10)
        ax.set_ylabel("Wall Hit Rate (%)")
        ax.set_title(f"{VARIANT_LABELS[mv]} — Wall Hit Rate")
        ax.grid(axis="y", alpha=0.3)

        # Dead-end Dwell Time
        ax = axes[row][1]
        means = [results[mv][sig]["dead_end_dwell"]     if results[mv][sig] else 0 for sig in SIGNALS]
        stds  = [results[mv][sig]["dead_end_dwell_std"] if results[mv][sig] else 0 for sig in SIGNALS]
        ax.bar(x, means, color=colors, edgecolor="white", yerr=stds, capsize=5, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sig_labels, fontsize=10)
        ax.set_ylabel("Avg Dead-end Dwell Steps")
        ax.set_title(f"{VARIANT_LABELS[mv]} — Dead-end Dwell Time")
        ax.grid(axis="y", alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SIGNAL_COLORS[s], label=SIGNAL_LABELS[s]) for s in SIGNALS
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "maze_variants_eval.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"图表已保存: {out}")


def save_npz(results: Dict):
    rows = []
    for mv in MAZE_VARIANTS:
        for sig in SIGNALS:
            d = results[mv].get(sig)
            if d:
                rows.append({
                    "variant": mv,
                    "signal":  sig,
                    "wall_hit_rate":  d["wall_hit_rate"],
                    "wall_hit_std":   d["wall_hit_std"],
                    "dead_end_dwell": d["dead_end_dwell"],
                })

    if not rows:
        return

    npz_out = FIGURES_DIR / "maze_variants_eval.npz"
    np.savez(
        npz_out,
        variants=np.array([r["variant"]        for r in rows]),
        signals=np.array([r["signal"]          for r in rows]),
        wall_hit_rate=np.array([r["wall_hit_rate"]  for r in rows]),
        dead_end_dwell=np.array([r["dead_end_dwell"] for r in rows]),
    )
    print(f"数值结果已保存: {npz_out}")


def main():
    print("=== 迷宫变体实验评估 ===\n")
    results = collect_results()
    plot_results(results)
    save_npz(results)


if __name__ == "__main__":
    main()
