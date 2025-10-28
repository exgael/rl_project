from typing import List
import random

from stable_baselines3 import PPO
from src.environment import run_episode
from src.dataclass import EpisodeBatch, EpisodeData

def evaluate(
    model: PPO,
    env_name: str,
    n_episodes: int = 100
) -> EpisodeBatch:
    """
    Evaluate model and collect all episode data.
    
    Returns:
        EpisodeBatch with all episodes and computed statistics
    """
    episodes: List[EpisodeData] = []
    
    for _ in range(n_episodes):
        seed: int = random.randint(0, 2**31 - 2)
        episode_data: EpisodeData = run_episode(
            model=model,
            env_name=env_name,
            seed=seed,
            deterministic=False # Evaluation on a stochastic policy
        )
        episodes.append(episode_data)
    
    return EpisodeBatch(episodes=tuple(episodes))