"""PPO 训练入口。"""

from __future__ import annotations

import os
import random

# macOS/conda 下常见 OpenMP 重复加载问题的兜底处理。
# 这是兼容性 workaround，优先保证训练脚本可运行。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

from config import Config
from envs.maze_env import MazeEnv
from models.cnn import MazeCNN


class IterationPrintCallback(BaseCallback):
    """每个 rollout 结束后打印一次关键训练指标。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.iteration = 0
        self._rollout_rewards = []
        self._rollout_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._rollout_rewards.append(float(ep.get("r", 0.0)))
                self._rollout_lengths.append(float(ep.get("l", 0.0)))
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        rollout_rewards = self.model.rollout_buffer.rewards
        mean_step_reward = float(np.mean(rollout_rewards)) if rollout_rewards.size > 0 else 0.0

        if self._rollout_rewards:
            mean_r = float(np.mean(self._rollout_rewards))
            mean_l = float(np.mean(self._rollout_lengths))
            print(
                f"[iter {self.iteration:04d}] "
                f"timesteps={self.num_timesteps} "
                f"mean_step_reward={mean_step_reward:.4f} "
                f"mean_ep_reward={mean_r:.3f} "
                f"mean_ep_len={mean_l:.1f}"
            )
        else:
            print(
                f"[iter {self.iteration:04d}] "
                f"timesteps={self.num_timesteps} "
                f"mean_step_reward={mean_step_reward:.4f} "
                "mean_ep_reward=N/A"
            )
        self._rollout_rewards.clear()
        self._rollout_lengths.clear()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: Config | None = None, seed: int = 42):
    """执行 PPO + CnnPolicy 训练。"""
    cfg = cfg or Config()

    _set_global_seed(seed)

    run_log_dir = os.path.join(cfg.log_dir, cfg.reward_type, f"seed_{seed}")
    run_save_path = os.path.join(cfg.save_path, cfg.reward_type, f"seed_{seed}")

    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_save_path, exist_ok=True)

    # 并行训练环境
    vec_env = make_vec_env(
        env_id=MazeEnv,
        n_envs=cfg.n_envs,
        env_kwargs={"cfg": cfg},
        seed=seed,
    )
    vec_env = VecTransposeImage(vec_env)

    # 独立评估环境
    eval_env = make_vec_env(
        env_id=MazeEnv,
        n_envs=1,
        env_kwargs={"cfg": cfg},
        seed=seed + 100,
    )
    eval_env = VecTransposeImage(eval_env)

    policy_kwargs = {
        "features_extractor_class": MazeCNN,
        "features_extractor_kwargs": {"features_dim": cfg.cnn_features_dim},
    }

    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1,
        tensorboard_log=run_log_dir,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=run_save_path,
        log_path=run_log_dir,
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=run_save_path,
        name_prefix=f"ppo_{cfg.reward_type}",
    )
    iter_print_callback = IterationPrintCallback()

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_callback, checkpoint_callback, iter_print_callback],
        progress_bar=True,
    )

    final_path = os.path.join(run_save_path, f"ppo_{cfg.reward_type}_final")
    model.save(final_path)
    print(f"训练完成，模型已保存到: {final_path}")


if __name__ == "__main__":
    seeds = [42, 43, 44]
    for seed in seeds:
        train(Config(), seed=seed)
