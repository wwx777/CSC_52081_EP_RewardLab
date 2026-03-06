"""项目全局配置。"""

from dataclasses import dataclass


@dataclass
class Config:
    """训练与环境配置。"""

    # 可选: immediate / accumulated_delay / fully_delayed / sparse
    reward_type: str = "immediate"
    maze_size: int = 11
    cell_size: int = 8  # 每格渲染像素
    max_steps: int = 300

    total_timesteps: int = 200_000
    n_envs: int = 4
    n_steps: int = 128
    batch_size: int = 256
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    cnn_features_dim: int = 128

    log_dir: str = "logs/"
    save_path: str = "checkpoints/"
