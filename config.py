"""项目全局配置。"""

from dataclasses import dataclass


@dataclass
class Config:
    """训练与环境配置。"""

    # 研究线1: signal_sparse / signal_euclidean_immediate / signal_bfs_immediate
    # 研究线2: timing_immediate / timing_accumulated_delay / timing_fully_delayed
    reward_type: str = "signal_bfs_immediate"
    maze_size: int = 11
    cell_size: int = 4  # 每格渲染像素
    max_steps: int = 200

    total_timesteps: int = 2_000_000
    n_envs: int = 8
    n_steps: int = 512
    batch_size: int = 256
    learning_rate: float = 3e-4
    ent_coef: float = 0.02
    cnn_features_dim: int = 128

    log_dir: str = "logs/"
    save_path: str = "checkpoints/"
