# rewardlab

基于 `stable-baselines3` 的迷宫强化学习项目，使用 `PPO + CnnPolicy`。

## 特性

- 迷宫环境：RGB 观测、离散动作（上/下/左/右）
- 奖励可插拔：抽象基类 + 注册表工厂
- 支持奖励类型：
  - `immediate`
  - `accumulated_delay`
  - `fully_delayed`
  - `sparse`

## 项目结构

```text
rewardlab/
├── config.py
├── config.yaml
├── train.py
├── eval.py
├── envs/
├── rewards/
├── models/
├── requirement.txt
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirement.txt
```

## 训练

```bash
python train.py
```

如果在 macOS 上出现 `OMP: Error #15`（`libomp` 重复初始化），可以先这样运行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python train.py
```

## 评估

```bash
python eval.py
```

## 切换奖励

只需要修改 `config.py` 或 `config.yaml` 里的 `reward_type`，例如：

```yaml
reward_type: fully_delayed
```
