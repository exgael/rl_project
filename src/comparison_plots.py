"""Generate comparison plots between experiments."""

from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    experiments: List[Dict[str, Any]],
    output_path: str,
    metric: str = "success_rate"
) -> None:
    """Plot learning curves comparing multiple experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # type: ignore
    fig.suptitle("Training Comparison: Direct vs Curriculum", fontsize=16)  # type: ignore
    
    metrics_map = {
        "success_rate": ("Success Rate", 0),
        "avg_episode_length": ("Average Episode Length", 1),
        "avg_reward": ("Average Reward", 2),
        "entropy": ("Entropy Coefficient", 3)
    }
    
    for metric_name, (label, idx) in metrics_map.items():
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]  # type: ignore
        
        for exp in experiments:
            name = exp['config'].get('experiment_name', 'Unknown')
            metrics = exp['metrics']
            
            steps = [m['step'] for m in metrics]
            values = [m[metric_name] for m in metrics]
            
            ax.plot(steps, values, label=name, linewidth=2, marker='o', markersize=4)  # type: ignore
        
        ax.set_xlabel('Training Steps', fontsize=12)  # type: ignore
        ax.set_ylabel(label, fontsize=12)  # type: ignore
        ax.set_title(label, fontsize=14)  # type: ignore
        ax.legend(fontsize=10)  # type: ignore
        ax.grid(True, alpha=0.3)  # type: ignore
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # type: ignore
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def plot_curriculum_stages(
    curriculum_exp: Dict[str, Any],
    output_path: str
) -> None:
    """Plot curriculum learning stages and transitions."""
    metrics = curriculum_exp['metrics']
    
    stages: List[str] = []
    stage_steps: List[int] = []
    stage_success: List[Tuple[int, float, str]] = []
    
    current_stage = None
    for m in metrics:
        stage = m.get('stage')
        if stage and stage != current_stage:
            if current_stage:
                stages.append(current_stage)
                stage_steps.append(m['step'])
            current_stage = stage
        
        if stage:
            stage_success.append((m['step'], m['success_rate'], stage))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))  # type: ignore
    fig.suptitle("Curriculum Learning Progress", fontsize=16)  # type: ignore
    
    unique_stages = sorted(set(s for _, _, s in stage_success))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_stages)))  # type: ignore
    stage_colors: Dict[str, Any] = {stage: colors[i] for i, stage in enumerate(unique_stages)}
    
    for stage in unique_stages:
        stage_data = [(step, sr) for step, sr, s in stage_success if s == stage]
        if stage_data:
            steps, success_rates = zip(*stage_data)
            ax1.plot(steps, success_rates, label=stage, linewidth=2, color=stage_colors[stage])  # type: ignore
    
    for step in stage_steps:
        ax1.axvline(step, color='red', linestyle='--', alpha=0.5)  # type: ignore
    
    ax1.set_xlabel('Training Steps', fontsize=12)  # type: ignore
    ax1.set_ylabel('Success Rate', fontsize=12)  # type: ignore
    ax1.set_title('Success Rate by Stage', fontsize=14)  # type: ignore
    ax1.legend(fontsize=8, loc='best')  # type: ignore
    ax1.grid(True, alpha=0.3)  # type: ignore
    
    stage_avg_steps: List[int] = []
    stage_names: List[str] = []
    prev_step = 0
    for i, stage in enumerate(stages):
        if i < len(stage_steps):
            stage_avg_steps.append(stage_steps[i] - prev_step)
            stage_names.append(stage.split('-')[-1])
            prev_step: int = stage_steps[i]
    
    if stage_avg_steps:
        ax2.bar(range(len(stage_avg_steps)), stage_avg_steps, color='steelblue')  # type: ignore
        ax2.set_xticks(range(len(stage_names)))  # type: ignore
        ax2.set_xticklabels(stage_names, rotation=45, ha='right')  # type: ignore
        ax2.set_xlabel('Stage', fontsize=12)  # type: ignore
        ax2.set_ylabel('Steps to Master', fontsize=12)  # type: ignore
        ax2.set_title('Training Steps per Stage', fontsize=14)  # type: ignore
        ax2.grid(True, alpha=0.3, axis='y')  # type: ignore
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # type: ignore
    plt.close()
    print(f"Curriculum stages plot saved: {output_path}")


def generate_summary_statistics(
    experiments: List[Dict[str, Any]],
    output_path: str
) -> None:
    """Generate summary statistics comparison."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for exp in experiments:
            name = exp['config'].get('experiment_name', 'Unknown')
            metrics = exp['metrics']
            
            if not metrics:
                continue
            
            final_metric = metrics[-1]
            total_steps = final_metric['step']
            final_success = final_metric['success_rate']
            
            success_90_step = None
            for m in metrics:
                if m['success_rate'] >= 0.90:
                    success_90_step = m['step']
                    break
            
            f.write(f"Experiment: {name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Steps: {total_steps:,}\n")
            f.write(f"  Final Success Rate: {final_success:.2%}\n")
            
            if success_90_step:
                f.write(f"  Steps to 90% Success: {success_90_step:,}\n")
            else:
                f.write(f"  Steps to 90% Success: Not reached\n")
            
            f.write(f"  Final Avg Episode Length: {final_metric['avg_episode_length']:.1f}\n")
            f.write(f"  Final Avg Reward: {final_metric['avg_reward']:.2f}\n")
            f.write("\n")
    
    print(f"Summary statistics saved: {output_path}")

