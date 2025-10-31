"""Experiment logging functions for training runs."""

import json
import os
from typing import Dict, Any
from dataclasses import asdict
from datetime import datetime
import pandas as pd

from src.dataclass import EpisodeBatch, EvaluationRecord

class FileManager:
    """Tracks training progress and saves results in a directory."""
    
    def __init__(self, experiment_name: str, output_dir: str = "results") -> None:
        self.run_dir: str = self.__create_dirs(experiment_name, output_dir)

    def __create_dirs(self, experiment_name: str, output_dir: str) -> str:
        """Create experiment directory structure and return run_dir path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)
        
        print(f"Experiment directory initialized: {run_dir}")
        return run_dir

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuration saved: {os.path.join(self.run_dir, "config.json")}")

    def log_env_allocation(self, allocation: Dict[str, int]) -> None:
        """Log environment allocation."""
        with open(os.path.join(self.run_dir, "env_allocation.json"), "w") as f:
            json.dump(allocation, f, indent=4)
        print(f"✓ Environment allocation saved: {os.path.join(self.run_dir, "env_allocation.json")}")
    
    def dump_eval_to_csv(
        self,
        total_step: int,
        stage: str,
        stage_step: int,
        batch: EpisodeBatch,
        model: Any,
        allocation: Dict[str, int]
    ) -> None:
        """Append evaluation record to CSV file."""
        record = EvaluationRecord(
            total_step=total_step,
            stage=stage,
            stage_step=stage_step,
            n_episodes=batch.n_episodes,
            success_rate=batch.success_rate,
            mean_reward=batch.mean_reward,
            std_reward=batch.std_reward,
            min_reward=batch.min_reward,
            max_reward=batch.max_reward,
            mean_length=batch.mean_length,
            std_length=batch.std_length,
            min_length=batch.min_length,
            max_length=batch.max_length,
            entropy_coef=float(model.ent_coef),
            learning_rate=float(model.learning_rate),
            clip_range=float(model.clip_range(1.0)),
            mean_entropy=float(batch.mean_entropy),
            allocation=allocation
        )

        # Convert to dict and serialize allocation to JSON string
        record_dict = asdict(record)
        record_dict['allocation'] = json.dumps(allocation)
        
        
        csv_path = os.path.join(self.run_dir, "evaluations.csv")
        df = pd.DataFrame([record_dict])
        
        # Append to existing CSV or create new one with header
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)
        
        print(f"    → Evaluation saved to: {csv_path}")
    
    def save_checkpoint(self, model: Any, stage: str, total_step: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.run_dir, "checkpoints", f"{stage}_step_{total_step}.zip")
        model.save(checkpoint_path)
        print(f"    → Checkpoint saved: {checkpoint_path}")
    
    def get_visualization_dir(self) -> str:
        """Get directory for visualizations."""
        return os.path.join(self.run_dir, "visualizations")