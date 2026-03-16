# CSC 52081 EP — Reward Lab

PPO + CNN 在随机迷宫上的奖励设计研究。通过两条独立研究线，系统分析奖励函数的**信号质量**与**发放时机**对 RL 学习的影响。

---

## 研究设计

### 研究线 1：信号质量（Signal Quality）
> 相同的发放时机（即时），改变距离信号的准确程度

| 注册名 | 文件 | 距离度量 | 奖励公式 |
|--------|------|---------|---------|
| `signal_sparse` | `rewards/signal_sparse.py` | 无 | `−0.01/step + 20.0 at goal` |
| `signal_euclidean_immediate` | `rewards/signal_euclidean_immediate.py` | 欧氏直线距离（穿墙） | `1.0×Δd_euclid − 0.01 + 20.0 at goal` |
| `signal_dfs_immediate` | `rewards/signal_dfs_immediate.py` | DFS 树深度（迷宫内，非最短路） | `1.0×Δd_dfs − 0.01 + 20.0 at goal` |
| `signal_bfs_immediate` | `rewards/signal_bfs_immediate.py` | BFS 真实最短路 | `1.0×Δd_bfs − 0.01 + 20.0 at goal` |

**共用参数：**

| 参数 | 值 | 说明 |
|------|----|------|
| `progress_multiplier` | 1.0 | 距离进步量的缩放系数 |
| `step_penalty` | −0.01 | 每步固定扣分（含撞墙步） |
| `goal_bonus` | +20.0 | 到达终点的额外奖励 |
| 撞墙惩罚 | 0（无额外扣分） | 撞墙时位置不变 → Δd=0 → 只扣 step_penalty |

**各信号的梯度特性：**
- `sparse`：无梯度，agent 只能靠探索碰运气找到终点
- `euclidean`：梯度穿墙，靠近终点的直线方向可能被墙阻断，产生误导
- `dfs`：迷宫内真实距离，但 DFS 树非最短路，可能绕远
- `bfs`：最优距离梯度，每步都朝真实最短路方向引导

### 研究线 2：发放时机（Temporal Structure）
> 相同的信号（BFS 最短路距离差），改变奖励的发放频率

| 注册名 | 文件 | 发放时机 | 说明 |
|--------|------|---------|------|
| `timing_immediate` | `rewards/timing_immediate.py` | 每步 | 与 `signal_bfs_immediate` 完全相同 |
| `timing_accumulated_delay` | `rewards/timing_accumulated_delay.py` | 每 10 步批量发放 | 累积 10 步的 BFS progress 一次性给出 |
| `timing_fully_delayed` | `rewards/timing_fully_delayed.py` | episode 结束时发放 | 整个 episode 的累积奖励在终止时一次性给出 |

**共用参数（与研究线1一致）：**
`progress_multiplier=1.0`，`step_penalty=−0.01`，`goal_bonus=+20.0`，信号均为 BFS 距离差

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

## 扩展实验（新增）

### 第一类：Post-hoc Evaluation（用现有 checkpoint，无需重训）

> **前提**：主实验已完成，`checkpoints/` 下存在各 reward 的模型。

```bash
# 路径最优性比率：actual_steps / BFS_shortest（约 10 分钟）
python analysis/eval_path_optimality.py

# Value gradient 方向对齐分析：撞墙率 & BFS 对齐率（约 30-60 分钟）
python analysis/eval_gradient_alignment.py
```

| 脚本 | 输出文件 | 指标说明 |
|------|----------|---------|
| `analysis/eval_path_optimality.py` | `path_optimality.png` / `.npz` | **Path Optimality Ratio** = 实际步数 / BFS最短路，越接近1越好 |
| `analysis/eval_gradient_alignment.py` | `gradient_alignment.png` / `.npz` | **Wall Misalignment Rate**（gradient指向墙的比例）& **BFS Alignment Rate**（gradient方向与BFS最优方向一致的比例） |

---

### 第二类：迷宫变体新实验（需重新训练）

实验矩阵：`2 变体 × 2 信号 × 3 seeds = 12 次训练`

| 变体 | 生成算法 | 特点 |
|------|---------|------|
| `dead_end_dense` | 随机 Prim 算法 | 大量短分支，dead-end 密集 |
| `long_corridor` | 方向偏置回溯 | 偏向直线延伸，长走廊 |

信号：`signal_euclidean_immediate`、`signal_bfs_immediate`（timing 固定为 immediate）

```bash
# Step 1：启动 12 个训练任务（自动并行到 6 张 GPU）
python run_maze_variants.py

# Step 2：训练完成后评估
python analysis/eval_maze_variants.py
```

- checkpoint 保存：`checkpoints_variants/{variant}_{signal}/seed_{seed}/`
- logs 保存：`logs_variants/{variant}_{signal}/seed_{seed}/`

| 脚本 | 输出文件 | 指标说明 |
|------|----------|---------|
| `analysis/eval_maze_variants.py` | `maze_variants_eval.png` / `.npz` | **Wall Hit Rate** & **Dead-end Dwell Time**（每次进入 dead-end 的平均停留步数） |

---

## 项目结构（含新增文件）

```
├── config.py                        # 全局配置（新增 maze_variant / run_name 字段）
├── train.py                         # PPO 训练函数（含 ExplainedVarianceCallback）
├── eval.py                          # 模型评估与 GIF 生成
├── run_main.py                      # 一键运行主实验
├── run_maze_variants.py             # 【新增】迷宫变体实验训练入口（12个任务）
├── run_analysis.py                  # 一键运行分析实验
├── envs/
│   ├── maze_env.py                  # 迷宫环境（支持 maze_variant 参数）
│   └── maze_generator.py            # 迷宫生成（新增 dead_end_dense / long_corridor）
├── rewards/
│   ├── signal_sparse.py
│   ├── signal_euclidean_immediate.py
│   ├── signal_bfs_immediate.py
│   ├── timing_immediate.py
│   ├── timing_accumulated_delay.py
│   └── timing_fully_delayed.py
├── models/
│   └── cnn.py
└── analysis/
    ├── utils.py
    ├── plot_learning_curves.py
    ├── plot_success_rate.py
    ├── plot_sample_efficiency.py
    ├── plot_behavior.py
    ├── plot_value_heatmap.py
    ├── plot_explained_variance.py
    ├── eval_path_optimality.py      # 【新增】路径最优性 post-hoc 评估
    ├── eval_gradient_alignment.py   # 【新增】value gradient 对齐 post-hoc 评估
    └── eval_maze_variants.py        # 【新增】迷宫变体指标评估
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
