"""Custom Stable-Baselines3 feature extractor for multi-timeframe data."""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiTFExtractor(BaseFeaturesExtractor):
    """1D CNN + LSTM extractor with attention-based fusion across streams."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 4:
            raise ValueError("Expected 4D observation space (assets, timeframes, window, features)")

        self.num_assets, self.num_timeframes, self.window_size, self.feature_dim = observation_space.shape
        self.num_streams = self.num_assets * self.num_timeframes

        self.cnn = nn.Sequential(
            nn.Conv1d(self.feature_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, batch_first=True)
        self.attention = nn.Linear(lstm_hidden, 1)
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        streams = observations.view(batch_size, self.num_streams, self.window_size, self.feature_dim)
        streams = streams.reshape(batch_size * self.num_streams, self.window_size, self.feature_dim)

        # CNN expects (batch, channels, time)
        x = streams.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = last_hidden.view(batch_size, self.num_streams, -1)

        attn_scores = self.attention(last_hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        fused = torch.sum(last_hidden * attn_weights, dim=1)

        return self.projection(fused)


__all__ = ["MultiTFExtractor"]

