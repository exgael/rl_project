import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
import re

from src.environment import create_env
from src.dataclass import EpisodeData, MINIGRID_ACTION_NAMES


def get_action_probs(model: BaseAlgorithm, obs: np.ndarray) -> Tuple[np.ndarray, int]:
    """Get action probabilities from model."""
    if obs.ndim == 3 and obs.shape[2] == 3:
        obs = obs.transpose(2, 0, 1)
    
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        action_logits = model.policy.get_distribution(obs_tensor).distribution.logits # type: ignore
        action_probs = F.softmax(action_logits, dim=1).squeeze().cpu().numpy() # type: ignore
        predicted_action = int(action_logits.argmax(dim=1).item()) # type: ignore
    
    return action_probs, predicted_action

def visualize_eval_episode(
    model: BaseAlgorithm,
    episode: EpisodeData,
    timestep: int,
    output_dir: str,
) -> None:
    """Visualize episode using recorded trajectory (not replayed)."""
    env = create_env(episode.env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=episode.seed)
    
    frames: List[np.ndarray] = []
    action_probs_list: List[np.ndarray] = []
    
    # Use recorded actions, don't re-infer
    for _, action in enumerate(episode.actions):
        action_probs, _ = get_action_probs(model, obs)
        frame: np.ndarray = env.render()  # type: ignore
        
        frames.append(frame)
        action_probs_list.append(action_probs)
        
        obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore
        if terminated or truncated:
            break
    
    env.close()
    
    # Rest of visualization code...
    num_frames = min(len(frames), 8)
    if num_frames == 0:
        return
    
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    
    status = "✓ Success" if episode.success else ("⊗ Truncated" if episode.truncated else "✗ Failed")
    
    fig = plt.figure(figsize=(3 * num_frames, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, num_frames)
    fig.suptitle(
        f"{episode.env_name} @ {timestep:,} steps | "
        f"Length: {episode.episode_length} | "
        f"Reward: {episode.total_reward:.1f} | "
        f"Entropy: {episode.mean_entropy:.2f} | "
        f"{status}",
        fontsize=16
    )
    
    for idx, frame_idx in enumerate(indices):
        col = idx
        
        frame = frames[int(frame_idx)]
        action_probs = action_probs_list[int(frame_idx)]
        action = episode.actions[int(frame_idx)]  # Use recorded action
        
        ax_frame = fig.add_subplot(gs[0, col])
        ax_frame.imshow(frame)
        ax_frame.set_title(f"Step {frame_idx}", fontsize=9)
        ax_frame.axis('off')
        
        ax_action = fig.add_subplot(gs[1, col])
        action_names = [MINIGRID_ACTION_NAMES[i] for i in range(len(action_probs))]
        colors = ['green' if i == action else 'steelblue' for i in range(len(action_probs))]
        
        ax_action.barh(action_names, action_probs, color=colors)
        ax_action.set_xlim(0, 1)
        ax_action.set_xlabel('Prob', fontsize=6)
        ax_action.tick_params(axis='both', labelsize=5)
        
        for i, prob in enumerate(action_probs):
            if prob > 0.05:
                ax_action.text(prob + 0.02, i, f'{prob:.2f}', va='center', fontsize=5)
    
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize stage name for filename
    stage_safe = re.sub(r'[^a-zA-Z0-9_-]', '', episode.env_name.replace('-', '_'))
    output_path = os.path.join(output_dir, f"eval_{timestep}_{stage_safe}.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"    → Saved visualization: {output_path}")
