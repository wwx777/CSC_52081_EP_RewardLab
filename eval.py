"""模型评估与单回合可视化。"""

from __future__ import annotations

import os
from typing import Optional

# macOS/conda 下常见 OpenMP 重复加载问题的兜底处理。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from config import Config
from envs.maze_env import MazeEnv


def evaluate(model_path: str, cfg: Config, n_episodes: int = 10):
    """加载模型并统计成功率与平均步数。"""
    env = MazeEnv(cfg)
    model = PPO.load(model_path)

    success = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        total_steps += steps
        success += int(bool(info.get("reached_goal", False)))

    success_rate = success / n_episodes
    avg_steps = total_steps / n_episodes

    print(f"评估回合数: {n_episodes}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均步数: {avg_steps:.2f}")


def render_episode(model_path: str, cfg: Config, save_gif: Optional[str] = None):
    """运行一个回合并展示/保存动画。"""
    env = MazeEnv(cfg)
    model = PPO.load(model_path)

    obs, _ = env.reset(seed=2026)
    frames = [obs]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(obs)
        done = terminated or truncated

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=80, blit=True)

    if save_gif:
        ani.save(save_gif, writer="pillow", fps=12)
        print(f"GIF 已保存到: {save_gif}")
    else:
        plt.show()


if __name__ == "__main__":
    cfg = Config()
    default_model = f"{cfg.save_path}/ppo_{cfg.reward_type}_final"
    evaluate(default_model, cfg, n_episodes=5)
