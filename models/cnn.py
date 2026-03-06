"""自定义 CNN 特征提取器。"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MazeCNN(BaseFeaturesExtractor):
    """用于迷宫 RGB 图像的 CNN 特征提取网络。"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 用 dummy forward 自动推断卷积输出维度。
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            sample = self._to_chw(sample) / 255.0
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError(f"期望 4D 输入，实际形状: {tuple(observations.shape)}")
        x = self._to_chw(observations.float()) / 255.0
        return self.linear(self.cnn(x))

    @staticmethod
    def _to_chw(x: torch.Tensor) -> torch.Tensor:
        """将输入统一到 (B, C, H, W)。兼容 CHW 与 HWC。"""
        if x.shape[1] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2)
        raise ValueError(f"无法识别通道维，输入形状: {tuple(x.shape)}")
