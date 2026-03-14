"""模型评估与单回合可视化。"""

from __future__ import annotations

import os
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from config import Config
from envs.maze_env import MazeEnv


def make_eval_env(cfg: Config, seed: int = 123):
    env = make_vec_env(
        env_id=MazeEnv,
        n_envs=1,
        env_kwargs={"cfg": cfg},
        seed=seed,
    )
    env = VecTransposeImage(env)
    return env


def evaluate(model_path: str, cfg: Config, n_episodes: int = 10):
    env = make_eval_env(cfg, seed=123)
    model = PPO.load(model_path)

    success = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs = env.reset()

        if ep == 0:
            print("obs shape:", obs.shape)
            print("obs dtype:", obs.dtype)
            print("obs min/max:", np.min(obs), np.max(obs))

        done = False
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            done = bool(dones[0])
            last_info = infos[0]
            steps += 1

        total_steps += steps
        success += int(bool(last_info.get("reached_goal", False)))

    success_rate = success / n_episodes
    avg_steps = total_steps / n_episodes

    print(f"评估回合数: {n_episodes}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均步数: {avg_steps:.2f}")

    env.close()


def render_episode(model_path: str, cfg: Config, save_gif: Optional[str] = None):
    env = make_eval_env(cfg, seed=2026)
    model = PPO.load(model_path)

    obs = env.reset()
    frames = []

    done = False
    step = 0

    while not done:
        # VecTransposeImage 后 obs 是 (1, C, H, W)
        # 为了可视化，要转回 HWC
        frame = np.transpose(obs[0], (1, 2, 0))
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        print(f"step {step}")
        print("action:", int(action[0]))
        print("agent_pos:", infos[0].get("agent_pos"), "reward:", float(rewards[0]))

        done = bool(dones[0])
        step += 1

    # 最后一帧
    frame = np.transpose(obs[0], (1, 2, 0))
    frames.append(frame)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=80,
        blit=True,
    )

    if save_gif:
        ani.save(save_gif, writer="pillow", fps=12)
        print(f"GIF 已保存到: {save_gif}")
    else:
        plt.show()

    env.close()


if __name__ == "__main__":
    cfg = Config()
    default_model = f"{cfg.save_path}/ppo_{cfg.reward_type}_final"
    evaluate(default_model, cfg, n_episodes=20)
    # render_episode(default_model, cfg)