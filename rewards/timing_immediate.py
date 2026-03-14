"""timing 研究线的即时奖励基准（与 signal_bfs_immediate 底层完全一致）。"""

from __future__ import annotations

from rewards.base_reward import register_reward
from rewards.signal_bfs_immediate import BFSImmediateReward


@register_reward("timing_immediate")
class TimingImmediateReward(BFSImmediateReward):
    """
    研究线2（发放时机）的基准组：BFS 距离差，每步立刻发放。
    直接复用 signal_bfs_immediate 的实现，保证底层信号完全一致。
    """
