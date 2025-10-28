from typing import Tuple, Dict, Any
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)  # type: ignore
        c = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 2), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.tensor(observation_space.sample()[None]).float()
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs.float()))

    def forward_with_activations(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.float()
        last = None
        for layer in self.cnn:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                last = x
        return self.linear(x), last  # type: ignore

def get_policy_kwargs() -> Dict[str, Any]:
    return {"features_extractor_class": MinigridFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 256}}