from typing import Callable, Dict, Optional, Any, List, Tuple
import numpy as np
import gymnasium as gym
import torch

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.dataclass import EpisodeData

def create_env(name: str, render_mode: Optional[str] = None) -> gym.Env[Any, Any]:
    """Create single MiniGrid environment with step penalty for efficiency."""
    assert isinstance(name, str), "name must be a string"
    assert len(name) > 0, "name must not be empty"
    env = gym.make(name, render_mode=render_mode)  # type: ignore
    env = ImgObsWrapper(env)
    return env


def make_vec_env(name: str, n_envs: int = 8) -> SubprocVecEnv:
    """Create vectorized environment for parallel training."""
    # Fix: Use proper closure to avoid serialization issues
    def _make_env_fn(env_name: str) -> Callable[[], gym.Env[Any, Any]]:
        """Closure factory to properly capture env_name."""
        def _init():
            return create_env(env_name)
        return _init
    
    env_fns: List[Callable[[], gym.Env[Any, Any]]] = [_make_env_fn(name) for _ in range(n_envs)]
    return SubprocVecEnv(env_fns)


def make_fixed_mixed_vec_env(
    allocation: Dict[str, int]
) -> SubprocVecEnv:
    """Create vectorized envs according to a fixed allocation"""
    # Flatten allocation into list of stages e.g. ["S3R1","S3R1","S3R2",...]
    env_assignments: List[str] = []
    for stage, count in allocation.items():
        env_assignments.extend([stage] * count)

    # Safety: ensure ordering stability
    stages = env_assignments

    # Create env factories
    def make_env_fn(stage: str) -> Callable[[], gym.Env[Any, Any]]:
        def _init() -> gym.Env[Any, Any]:
            return create_env(stage)
        return _init

    env_fns = [make_env_fn(stage) for stage in stages]

    env = SubprocVecEnv(env_fns)

    return env

def run_episode(
    model: PPO,
    env_name: str,
    seed: int,
    deterministic: bool = True,
    render_mode: Optional[str] = None,
) -> EpisodeData:
    """Run single episode with given seed and return summary."""
    env = create_env(env_name, render_mode=render_mode)
    
    obs, _ = env.reset(seed=seed)
    assert isinstance(obs, np.ndarray), "Observation must be a numpy ndarray"
    assert obs.ndim == 3, f"Expected 3D observation, got shape {obs.shape}" # type: ignore
    assert obs.shape[2] == 3, f"Expected 3 channels in last dim, got shape {obs.shape}" # type: ignore
    
    total_reward: float = 0.0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    entropies: List[float] = []
    actions: List[int] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)  # type: ignore
        action = int(action)  # ← Cast to int
        # Compute log prob variance for this state
        # MiniGrid ImgObsWrapper returns HWC format, transpose to CHW for PyTorch
   
        obs_chw: np.ndarray = obs.transpose(2, 0, 1) # type: ignore
        obs_tensor = torch.FloatTensor(obs_chw).unsqueeze(0).to(model.device)
        with torch.no_grad():
            try:
                distribution = model.policy.get_distribution(obs_tensor)
                entropy = distribution.entropy().item() # type: ignore
                entropies.append(entropy)
            except Exception as e:
                # If entropy computation fails, skip this step
                print(f"Warning: Failed to compute entropy: {e}")
        
        obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
        total_reward += float(reward)
        steps += 1
        actions.append(action)
    
    env.close()
    
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    
    return EpisodeData(
        seed=seed,
        env_name=env_name,
        total_reward=total_reward,
        episode_length=steps,
        terminated=terminated,
        truncated=truncated,
        success=terminated and total_reward > 0,
        mean_entropy=mean_entropy,
        actions=tuple(actions)
    )
