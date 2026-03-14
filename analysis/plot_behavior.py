"""分析图：各 reward 的行为分析——动作分布 & 撞墙率 & 探索覆盖率。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from config import Config
from envs.maze_env import MazeEnv
from analysis.utils import ALL_REWARDS, DISPLAY_NAMES, COLORS, FIGURES_DIR, best_model_path

ACTION_NAMES = ["Up", "Down", "Left", "Right"]
N_EPISODES = 50


def analyze_behavior(reward_type: str, seed: int = 42) -> dict | None:
    model_p = best_model_path(reward_type, seed)
    if model_p is None:
        print(f"  [skip] {reward_type}: 模型不存在")
        return None

    cfg = Config()
    cfg.reward_type = reward_type
    env = make_vec_env(MazeEnv, n_envs=1, env_kwargs={"cfg": cfg}, seed=1000)
    env = VecTransposeImage(env)
    model = PPO.load(str(model_p))

    all_actions, wall_hit_rates, visited_counts = [], [], []

    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        ep_actions, ep_wall_hits, prev_pos = [], 0, None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = bool(dones[0])
            a = int(action[0])
            ep_actions.append(a)
            curr_pos = infos[0].get("agent_pos")
            if prev_pos is not None and curr_pos == prev_pos:
                ep_wall_hits += 1
            prev_pos = curr_pos

        all_actions.extend(ep_actions)
        wall_hit_rates.append(ep_wall_hits / len(ep_actions) if ep_actions else 0)
        # visited = 访问过的不同格子数（从 info 无法直接拿，用 env 内部）
        visited_counts.append(infos[0].get("steps", len(ep_actions)))

    env.close()

    cnt = Counter(all_actions)
    total = len(all_actions)
    action_dist = [cnt.get(i, 0) / total for i in range(4)]

    return {
        "action_dist":    action_dist,
        "wall_hit_rate":  float(np.mean(wall_hit_rates)),
        "mean_ep_steps":  float(np.mean(visited_counts)),
    }


if __name__ == "__main__":
    results = {}
    for rt in ALL_REWARDS:
        print(f"分析 {rt} ...")
        r = analyze_behavior(rt)
        if r:
            results[rt] = r

    if not results:
        print("没有可用模型，请先运行 run_main.py")
        sys.exit(0)

    reward_types = list(results.keys())
    n = len(reward_types)
    x = np.arange(n)
    labels = [DISPLAY_NAMES[rt] for rt in reward_types]
    colors = [COLORS[rt] for rt in reward_types]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Behavior Analysis (Deterministic Policy, 50 Episodes)", fontsize=13)

    # --- 动作分布 ---
    ax = axes[0]
    width = 0.2
    for i, action_name in enumerate(ACTION_NAMES):
        vals = [results[rt]["action_dist"][i] * 100 for rt in reward_types]
        ax.bar(x + i * width, vals, width, label=action_name)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Action Frequency (%)")
    ax.set_title("Action Distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- 撞墙率 ---
    ax = axes[1]
    vals = [results[rt]["wall_hit_rate"] * 100 for rt in reward_types]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.set_ylabel("Wall Hit Rate (%)")
    ax.set_title("Wall Hit Rate")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- 平均步数 ---
    ax = axes[2]
    vals = [results[rt]["mean_ep_steps"] for rt in reward_types]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.axhline(y=Config().max_steps, color="red", linestyle="--", linewidth=1, label="max_steps")
    ax.set_ylabel("Mean Episode Steps")
    ax.set_title("Mean Episode Length")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "behavior_analysis.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"保存: {out}")
