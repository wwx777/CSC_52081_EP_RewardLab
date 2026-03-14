"""一键运行所有主实验：6种reward × 3 seeds。"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from config import Config
from train import train

# 研究线1：信号质量（唯一变量：距离度量方式）
SIGNAL_REWARDS = [
    "signal_sparse",
    "signal_euclidean_immediate",
    "signal_dfs_immediate",
    "signal_bfs_immediate",
]

# 研究线2：发放时机（唯一变量：何时发放，底层信号完全一致）
TIMING_REWARDS = [
    "timing_immediate",
    "timing_accumulated_delay",
    "timing_fully_delayed",
]

SEEDS = [42, 43, 44]

if __name__ == "__main__":
    all_rewards = SIGNAL_REWARDS + TIMING_REWARDS
    total = len(all_rewards) * len(SEEDS)
    count = 0

    for reward_type in all_rewards:
        for seed in SEEDS:
            count += 1
            print(f"\n{'='*60}")
            print(f"[{count}/{total}] reward={reward_type}  seed={seed}")
            print(f"{'='*60}")
            cfg = Config()
            cfg.reward_type = reward_type
            train(cfg, seed=seed)

    print("\n所有训练完成！")
