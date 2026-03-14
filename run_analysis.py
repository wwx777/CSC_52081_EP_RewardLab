"""一键运行所有分析实验，生成所有图表到 analysis/figures/。"""

import os
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

SCRIPTS = [
    # 主图
    ("analysis/plot_learning_curves.py",   "主图1&2：两条研究线学习曲线"),
    # 辅助分析
    ("analysis/plot_success_rate.py",      "分析1：成功率曲线"),
    ("analysis/plot_sample_efficiency.py", "分析2：样本效率柱状图"),
    ("analysis/plot_behavior.py",          "分析3：行为分析（动作分布/撞墙率）"),
    ("analysis/plot_value_heatmap.py",     "分析4：Value Function 热力图"),
    ("analysis/plot_explained_variance.py","分析5：Explained Variance 曲线"),
]

if __name__ == "__main__":
    print("=" * 60)
    print("开始运行所有分析实验")
    print("=" * 60)

    for script, desc in SCRIPTS:
        print(f"\n>>> {desc}")
        print(f"    运行 {script} ...")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"    [ERROR] {script} 执行失败（exitcode={result.returncode}），跳过继续")

    print("\n" + "=" * 60)
    print("所有分析完成！图表保存在 analysis/figures/")
    print("=" * 60)
