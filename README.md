# CSC 52081 EP — Reward Lab

PPO + CNN 在随机迷宫上的奖励设计研究。通过两条独立研究线，系统分析奖励函数的**信号质量**与**发放时机**对 RL 学习的影响。

---

## 研究设计

### 研究线 1：信号质量（Signal Quality）
> 相同的发放时机（即时），改变信号的准确程度

| 注册名 | 文件 | 说明 |
|--------|------|------|
| `signal_sparse` | `rewards/signal_sparse.py` | 无距离引导，只有终点奖励 |
| `signal_euclidean_immediate` | `rewards/signal_euclidean_immediate.py` | 欧氏距离 shaping（存在绕路陷阱） |
| `signal_bfs_immediate` | `rewards/signal_bfs_immediate.py` | BFS 真实最短路距离 shaping |

### 研究线 2：发放时机（Temporal Structure）
> 相同的信号（BFS 距离差），改变发放的时机

| 注册名 | 文件 | 说明 |
|--------|------|------|
| `timing_immediate` | `rewards/timing_immediate.py` | 每步立刻发放 |
| `timing_accumulated_delay` | `rewards/timing_accumulated_delay.py` | 每 10 步批量发放 |
| `timing_fully_delayed` | `rewards/timing_fully_delayed.py` | episode 结束才发放 |

所有 reward 参数对齐：`progress × 1.0 − 0.01/step + 20.0 at goal`

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键运行所有主实验（6种reward × 3 seeds）
python run_main.py

# 一键生成所有分析图表
python run_analysis.py
```

---

## 主实验

**入口**：`run_main.py`

- 6 种 reward × 3 seeds（42 / 43 / 44）= 18 次训练
- 每次训练结果保存在：
  - `logs/{reward_type}/seed_{seed}/` — 训练日志 & evaluations.npz & explained_variance.npz
  - `checkpoints/{reward_type}/seed_{seed}/` — 模型 checkpoint & best_model.zip

---

## 分析实验

**入口**：`run_analysis.py`（依赖主实验训练结果）

所有图表保存到 `analysis/figures/`

| 脚本 | 输出文件 | 说明 |
|------|----------|------|
| `analysis/plot_learning_curves.py` | `learning_curves_signal.png` `learning_curves_timing.png` | **主图**：两条研究线的 eval reward 学习曲线（mean ± std across seeds） |
| `analysis/plot_success_rate.py` | `success_rate_signal.png` `success_rate_timing.png` | 成功率曲线（与 reward 曲线分开看） |
| `analysis/plot_sample_efficiency.py` | `sample_efficiency.png` | 各 reward 首次成功所需 timestep 柱状图（样本效率对比） |
| `analysis/plot_behavior.py` | `behavior_analysis.png` | 确定性策略的动作分布 & 撞墙率 & 平均步数 |
| `analysis/plot_value_heatmap.py` | `value_heatmap.png` | 三种信号质量 reward 的 V(s) 热力图对比（解释欧氏距离陷阱） |
| `analysis/plot_explained_variance.py` | `explained_variance.png` | timing 研究线的 Explained Variance 曲线（衡量 credit assignment 质量） |

---

## 项目结构

```
├── config.py                        # 全局配置
├── train.py                         # PPO 训练函数（含 ExplainedVarianceCallback）
├── eval.py                          # 模型评估与 GIF 生成
├── run_main.py                      # 一键运行主实验
├── run_analysis.py                  # 一键运行分析实验
├── envs/
│   ├── maze_env.py                  # 迷宫环境（RGB 观测，离散动作）
│   └── maze_generator.py            # 随机迷宫生成（递归回溯）
├── rewards/
│   ├── signal_sparse.py             # 研究线1：稀疏奖励
│   ├── signal_euclidean_immediate.py# 研究线1：欧氏距离即时奖励
│   ├── signal_bfs_immediate.py      # 研究线1&2：BFS 即时奖励（共享基准）
│   ├── timing_immediate.py          # 研究线2：即时发放
│   ├── timing_accumulated_delay.py  # 研究线2：每10步发放
│   └── timing_fully_delayed.py      # 研究线2：episode结束发放
├── models/
│   └── cnn.py                       # CNN 特征提取器
└── analysis/
    ├── utils.py                     # 共用常量和数据加载工具
    ├── plot_learning_curves.py      # 主图
    ├── plot_success_rate.py
    ├── plot_sample_efficiency.py
    ├── plot_behavior.py
    ├── plot_value_heatmap.py
    └── plot_explained_variance.py
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

macOS 出现 `OMP: Error #15` 时：

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python run_main.py
```
