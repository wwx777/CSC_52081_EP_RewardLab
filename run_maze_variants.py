"""迷宫变体新实验训练入口。

实验矩阵：
  2 迷宫变体 (dead_end_dense, long_corridor)
× 2 奖励信号 (signal_euclidean_immediate, signal_bfs_immediate)
× 3 seeds    (42, 43, 44)
= 12 次训练

checkpoint 保存在:  checkpoints_variants/{run_name}/seed_{seed}/
logs 保存在:        logs_variants/{run_name}/seed_{seed}/

run_name 命名规则: variant_{maze_variant}_{signal_short}
  例: variant_dead_end_dense_euclidean
      variant_long_corridor_bfs
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 实验矩阵
MAZE_VARIANTS = ["dead_end_dense", "long_corridor"]
SIGNALS = ["signal_euclidean_immediate", "signal_bfs_immediate"]
SEEDS = [42, 43, 44]

# 简短名称映射（用于 run_name）
SIGNAL_SHORT = {
    "signal_euclidean_immediate": "euclidean",
    "signal_bfs_immediate":       "bfs",
}

NUM_GPUS = 6
JOBS_PER_GPU = 2   # 变体实验与原始实验并行，保守设 2


def make_run_name(maze_variant: str, signal: str) -> str:
    return f"variant_{maze_variant}_{SIGNAL_SHORT[signal]}"


def run_one(maze_variant: str, signal: str, seed: int, gpu_id: int):
    run_name = make_run_name(maze_variant, signal)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"

    cmd = [
        sys.executable, "-c",
        f"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
from config import Config
from train import train
cfg = Config()
cfg.reward_type   = '{signal}'
cfg.maze_variant  = '{maze_variant}'
cfg.run_name      = '{run_name}'
cfg.log_dir       = 'logs_variants/'
cfg.save_path     = 'checkpoints_variants/'
cfg.use_wandb     = False
train(cfg, seed={seed})
""",
    ]
    result = subprocess.run(cmd, env=env)
    return maze_variant, signal, seed, result.returncode


if __name__ == "__main__":
    jobs = [
        (mv, sig, seed)
        for mv in MAZE_VARIANTS
        for sig in SIGNALS
        for seed in SEEDS
    ]
    total = len(jobs)
    max_workers = NUM_GPUS * JOBS_PER_GPU

    print("=== 迷宫变体新实验 ===")
    print(f"共 {total} 个任务，{NUM_GPUS} 张 GPU × {JOBS_PER_GPU} 任务 = {max_workers} 并发\n")
    print("任务列表:")
    for mv, sig, seed in jobs:
        print(f"  {make_run_name(mv, sig)}  seed={seed}")
    print()

    completed = 0
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, mv, sig, seed, i % NUM_GPUS): (mv, sig, seed)
            for i, (mv, sig, seed) in enumerate(jobs)
        }

        for future in as_completed(futures):
            mv, sig, seed = futures[future]
            completed += 1
            try:
                _, _, _, rc = future.result()
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                if rc != 0:
                    failed.append((mv, sig, seed))
            except Exception as e:
                status = f"ERROR: {e}"
                failed.append((mv, sig, seed))
            run_name = make_run_name(mv, sig)
            print(f"[{completed}/{total}] {run_name} seed={seed} -> {status}")

    print("\n所有变体训练任务完成！")
    if failed:
        print(f"失败任务 ({len(failed)}):")
        for mv, sig, seed in failed:
            print(f"  {make_run_name(mv, sig)} seed={seed}")
    else:
        print("全部成功。")
        print("\n接下来运行评估:")
        print("  python analysis/eval_maze_variants.py")
