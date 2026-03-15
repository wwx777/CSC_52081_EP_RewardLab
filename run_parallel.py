"""并行运行所有主实验：4张 GPU 同时跑，每张 GPU 多个任务。"""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 研究线1：信号质量
SIGNAL_REWARDS = [
    "signal_sparse",
    "signal_euclidean_immediate",
    "signal_dfs_immediate",
    "signal_bfs_immediate",
]

# 研究线2：发放时机
TIMING_REWARDS = [
    "timing_immediate",
    "timing_accumulated_delay",
    "timing_fully_delayed",
]

SEEDS = [42, 43, 44]
NUM_GPUS = 6       # 使用 GPU 0~5
JOBS_PER_GPU = 4   # 每张 GPU 同时跑几个任务（L40 48GB 显存仅用 4%，轻松跑 3 个）


def run_one(reward_type: str, seed: int, gpu_id: int) -> tuple[str, int, int]:
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
cfg.reward_type = '{reward_type}'
cfg.use_wandb = True
train(cfg, seed={seed})
"""
    ]
    result = subprocess.run(cmd, env=env)
    return reward_type, seed, result.returncode


if __name__ == "__main__":
    all_rewards = SIGNAL_REWARDS + TIMING_REWARDS
    jobs = [(r, s) for r in all_rewards for s in SEEDS]
    total = len(jobs)

    max_workers = NUM_GPUS * JOBS_PER_GPU
    print(f"共 {total} 个任务，{NUM_GPUS} 张 GPU × {JOBS_PER_GPU} 任务 = {max_workers} 并发\n")

    completed = 0
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, reward, seed, i % NUM_GPUS): (reward, seed)
            for i, (reward, seed) in enumerate(jobs)
        }

        for future in as_completed(futures):
            reward, seed = futures[future]
            completed += 1
            try:
                _, _, rc = future.result()
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                if rc != 0:
                    failed.append((reward, seed))
            except Exception as e:
                status = f"ERROR: {e}"
                failed.append((reward, seed))
            print(f"[{completed}/{total}] {reward} seed={seed} -> {status}")

    print("\n所有任务完成！")
    if failed:
        print(f"失败任务 ({len(failed)}):")
        for r, s in failed:
            print(f"  reward={r} seed={s}")
