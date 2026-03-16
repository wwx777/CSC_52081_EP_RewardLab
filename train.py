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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

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
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "return/mean": mean_r,
                    "return/min": float(np.min(self._rollout_rewards)),
                    "return/max": float(np.max(self._rollout_rewards)),
                    "return/std": float(np.std(self._rollout_rewards)),
                    "ep_len/mean": mean_l,
                    "step_reward/mean": mean_step_reward,
                }, step=self.num_timesteps)
        else:
            print(
                f"[iter {self.iteration:04d}] "
                f"timesteps={self.num_timesteps} "
                f"mean_step_reward={mean_step_reward:.4f} "
                "mean_ep_reward=N/A"
            )
        self._rollout_rewards.clear()
        self._rollout_lengths.clear()


class ExplainedVarianceCallback(BaseCallback):
    """记录每个 rollout 的 explained variance，用于分析 credit assignment 质量。"""

    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
        self._timesteps: list = []
        self._ev_values: list = []
        self._iteration = 0

    def _on_rollout_start(self) -> None:
        if self._iteration > 0:
            ev = self.model.logger.name_to_value.get("train/explained_variance")
            if ev is not None:
                self._timesteps.append(self.num_timesteps)
                self._ev_values.append(float(ev))
        self._iteration += 1

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._timesteps:
            np.savez(
                self.save_path,
                timesteps=np.array(self._timesteps),
                explained_variance=np.array(self._ev_values),
            )


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

    run_id = cfg.run_name if cfg.run_name else cfg.reward_type
    run_log_dir = os.path.join(cfg.log_dir, run_id, f"seed_{seed}")
    run_save_path = os.path.join(cfg.save_path, run_id, f"seed_{seed}")

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
        best_model_save_path=None,   # 不存 best_model，只存最终 ckpt
        log_path=run_log_dir,
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    iter_print_callback = IterationPrintCallback()
    ev_callback = ExplainedVarianceCallback(
        save_path=os.path.join(run_log_dir, "explained_variance.npz")
    )

    callbacks = [eval_callback, iter_print_callback, ev_callback]

    if cfg.use_wandb:
        assert _WANDB_AVAILABLE, "请先 pip install wandb"
        run = wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.reward_type}_seed{seed}",
            group=cfg.reward_type,
            config={
                "reward_type": cfg.reward_type,
                "seed": seed,
                "total_timesteps": cfg.total_timesteps,
                "n_envs": cfg.n_envs,
                "n_steps": cfg.n_steps,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "ent_coef": cfg.ent_coef,
            },
            sync_tensorboard=True,   # 自动同步 SB3 的 tensorboard 指标
            save_code=False,
            reinit=True,
        )
        callbacks.append(WandbCallback(verbose=0))

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(run_save_path, f"ppo_{run_id}_final")
    model.save(final_path)
    print(f"训练完成，模型已保存到: {final_path}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    seeds = [42, 43, 44]
    for seed in seeds:
        train(Config(), seed=seed)
