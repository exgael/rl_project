from dataclasses import dataclass, field
from typing import Tuple, Dict
import numpy as np

MINIGRID_ACTION_NAMES: Dict[int, str] = {
    0: "Left",
    1: "Right", 
    2: "Forward",
    3: "Pickup",
    4: "Drop",
    5: "Toggle",
    6: "Done"
}


@dataclass(frozen=True)
class EpisodeData:
    """Episode summary - can be replayed using checkpoint + seed."""
    seed: int
    env_name: str
    total_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    success: bool
    mean_entropy: float
    actions: Tuple[int, ...]
    
    @property
    def done(self) -> bool:
        """Episode finished (either terminated or truncated)."""
        return self.terminated or self.truncated

    def __str__(self) -> str:
        return f"EpisodeData(seed={self.seed}, env_name={self.env_name}, total_reward={self.total_reward}, episode_length={self.episode_length}, terminated={self.terminated}, truncated={self.truncated}, success={self.success}, mean_entropy={self.mean_entropy})"

@dataclass(frozen=True)
class EpisodeBatch:
    """Batch of episodes with computed statistics."""
    episodes: Tuple[EpisodeData, ...]
    
    @property
    def n_episodes(self) -> int:
        """Number of episodes in batch."""
        return len(self.episodes)
    
    @property
    def success_rate(self) -> float:
        """Fraction of successful episodes."""
        return float(np.mean([ep.success for ep in self.episodes]))
    
    @property
    def mean_reward(self) -> float:
        """Average reward across all episodes."""
        return float(np.mean([ep.total_reward for ep in self.episodes]))
    
    @property
    def std_reward(self) -> float:
        """Standard deviation of reward across all episodes."""
        return float(np.std([ep.total_reward for ep in self.episodes]))
    
    @property
    def min_reward(self) -> float:
        """Minimum reward across all episodes."""
        return float(np.min([ep.total_reward for ep in self.episodes]))
    
    @property
    def max_reward(self) -> float:
        """Maximum reward across all episodes."""
        return float(np.max([ep.total_reward for ep in self.episodes]))
    
    @property
    def mean_length(self) -> float:
        """Average episode length across all episodes."""
        return float(np.mean([ep.episode_length for ep in self.episodes]))
    
    @property
    def std_length(self) -> float:
        """Standard deviation of episode length across all episodes."""
        return float(np.std([ep.episode_length for ep in self.episodes]))
    
    @property
    def min_length(self) -> int:
        """Minimum episode length across all episodes."""
        return int(np.min([ep.episode_length for ep in self.episodes]))
    
    @property
    def max_length(self) -> int:
        """Maximum episode length across all episodes."""
        return int(np.max([ep.episode_length for ep in self.episodes]))
    
    @property
    def mean_entropy(self) -> float:
        """Average log probability variance across all episodes."""
        return float(np.mean([ep.mean_entropy for ep in self.episodes]))

    def __str__(self) -> str:
        return f"EpisodeBatch(n_episodes={self.n_episodes}, success_rate={self.success_rate}, mean_reward={self.mean_reward}, std_reward={self.std_reward}, min_reward={self.min_reward}, max_reward={self.max_reward}, mean_length={self.mean_length}, std_length={self.std_length}, min_length={self.min_length}, max_length={self.max_length}, mean_entropy={self.mean_entropy})"

@dataclass
class EvaluationRecord:
    """Comprehensive evaluation record with all metrics."""
    total_step: int
    stage: str
    stage_step: int
    n_episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float
    std_length: float
    min_length: int
    max_length: int
    entropy_coef: float
    learning_rate: float
    clip_range: float
    mean_entropy: float
    allocation: Dict[str, int] = field(default_factory=dict)
