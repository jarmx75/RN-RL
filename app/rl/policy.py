"""Política CNN-LSTM multi-temporal para SB3."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiTemporalFeatureExtractor(BaseFeaturesExtractor):
    """Extractor que aplica CNN1D + LSTM para fusionar timeframes."""

    def __init__(self, observation_space, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        n_pairs, window, channels = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.attn = nn.Linear(128, 1)
        self.output = nn.Linear(128 * n_pairs, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # observations shape: (batch, pairs, window, features)
        obs = observations.permute(0, 1, 3, 2)  # (batch, pairs, features, window)
        pair_embeddings: List[torch.Tensor] = []
        for pair in range(obs.shape[1]):
            x = obs[:, pair]
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)
            weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)
            context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
            pair_embeddings.append(context)
        fused = torch.cat(pair_embeddings, dim=1)
        return torch.relu(self.output(fused))


class MultiTemporalCnnLstmPolicy(ActorCriticPolicy):
    """Política actor-crítico con extractor personalizado."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", MultiTemporalFeatureExtractor)
        kwargs.setdefault("net_arch", dict(pi=[256, 128], vf=[256, 128]))
        super().__init__(*args, **kwargs)
