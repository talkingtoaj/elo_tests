"""
Paper-ready visualization generation for ELO assessment framework.

This module provides functions to generate publication-quality figures
for the research paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from utils import ProbabilityModel
import os


def plot_probability_curves_comparison(
    models: Dict[str, ProbabilityModel],
    elo_diff_range: Tuple[int, int] = (-2000, 2000),
    output_path: Optional[str] = None
):
    """
    Plot probability curves for multiple models on same axes.
    
    Args:
        models: Dictionary of {name: ProbabilityModel}
        elo_diff_range: Range of ELO differences to plot
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    elo_diffs = np.linspace(elo_diff_range[0], elo_diff_range[1], 500)
    
    colors = plt.cm.tab10.colors
    
    for i, (name, model) in enumerate(models.items()):
        probs = [model.get_probability(d) for d in elo_diffs]
        ax.plot(elo_diffs, probs, label=name, color=colors[i % len(colors)], linewidth=2)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('ELO Difference (Question - Student)', fontsize=12)
    ax.set_ylabel('P(Correct)', fontsize=12)
    ax.set_title('Probability Model Comparison', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(elo_diff_range)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved probability curves to {output_path}")
    
    plt.show()
    return fig


def plot_convergence_distribution(
    results_by_model: Dict[str, List[int]],
    output_path: Optional[str] = None
):
    """
    Plot histogram of rounds to convergence for different models.
    
    Args:
        results_by_model: Dictionary of {model_name: [rounds_list]}
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors
    
    # Calculate statistics
    stats = {}
    for name, rounds_list in results_by_model.items():
        stats[name] = {
            'mean': np.mean(rounds_list),
            'std': np.std(rounds_list),
            'median': np.median(rounds_list)
        }
    
    # Plot histograms
    for i, (name, rounds_list) in enumerate(results_by_model.items()):
        ax.hist(rounds_list, bins=range(1, max(max(r) for r in results_by_model.values()) + 2),
                alpha=0.6, label=f"{name} (μ={stats[name]['mean']:.1f})", 
                color=colors[i % len(colors)])
    
    ax.set_xlabel('Rounds to Convergence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence distribution to {output_path}")
    
    plt.show()
    return fig


def plot_empirical_vs_fitted(
    binned_data,
    fitted_model: ProbabilityModel,
    output_path: Optional[str] = None
):
    """
    Plot empirical data points with fitted curve overlaid.
    
    Args:
        binned_data: DataFrame with bin_center, mean_success_rate columns
        fitted_model: Fitted ProbabilityModel
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot empirical points
    ax.scatter(binned_data['bin_center'], binned_data['mean_success_rate'],
               s=binned_data['count']/2, alpha=0.6, label='Empirical Data',
               color='blue', edgecolors='darkblue')
    
    # Plot fitted curve
    elo_diffs = np.linspace(binned_data['bin_center'].min() - 100,
                           binned_data['bin_center'].max() + 100, 200)
    fitted_probs = [fitted_model.get_probability(d) for d in elo_diffs]
    ax.plot(elo_diffs, fitted_probs, 'r-', linewidth=2, label='Fitted Model')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('ELO Difference (Question - Student)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Empirical Data vs Fitted Model', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved empirical vs fitted to {output_path}")
    
    plt.show()
    return fig


def plot_convergence_trajectories(
    trajectories: List[Dict],
    output_path: Optional[str] = None
):
    """
    Show example convergence paths.
    
    Args:
        trajectories: List of dicts with 'true_elo', 'guessed_elo', 'variance', 'round' keys
        output_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by trajectory
    n_trajectories = len(set(t['trajectory_id'] for t in trajectories))
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    for traj_id in range(n_trajectories):
        traj_data = [t for t in trajectories if t['trajectory_id'] == traj_id]
        traj_data.sort(key=lambda x: x['round'])
        
        ax1.plot([t['round'] for t in traj_data],
                [t['guessed_elo'] for t in traj_data],
                color=colors[traj_id], alpha=0.5, linewidth=1)
        
        ax2.plot([t['round'] for t in traj_data],
                [t['variance'] for t in traj_data],
                color=colors[traj_id], alpha=0.5, linewidth=1)
    
    # Add true ELO reference lines
    true_elos = set(t['true_elo'] for t in trajectories)
    for te in true_elos:
        ax1.axhline(y=te, color='red', linestyle='--', alpha=0.3)
    
    ax1.axhline(y=500, color='green', linestyle=':', alpha=0.5, label='±500 band')
    ax1.axhline(y=-500, color='green', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Guessed ELO', fontsize=12)
    ax1.set_title('ELO Convergence Paths', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Variance Decay', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence trajectories to {output_path}")
    
    plt.show()
    return fig


def plot_model_parameters_comparison(
    models: Dict[str, ProbabilityModel],
    output_path: Optional[str] = None
):
    """
    Bar chart comparing model parameters side by side.
    
    Args:
        models: Dictionary of {name: ProbabilityModel}
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = ['steepness_upper', 'steepness_lower', 'offset']
    x = np.arange(len(params))
    width = 0.8 / len(models)
    
    colors = plt.cm.tab10.colors
    
    for i, (name, model) in enumerate(models.items()):
        values = [getattr(model, p) for p in params]
        ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Parameter Comparison', fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(['Steepness (Hard)', 'Steepness (Easy)', 'Offset'])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter comparison to {output_path}")
    
    plt.show()
    return fig


def plot_convergence_by_starting_elo(
    results: List[Dict],
    output_path: Optional[str] = None
):
    """
    Plot rounds to convergence vs starting ELO.
    
    Args:
        results: List of {true_elo, initial_elo, rounds, converged}
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    true_elos = [r['true_elo'] for r in results]
    rounds = [r['rounds'] for r in results]
    converged = [r['converged'] for r in results]
    
    # Separate converged and not converged
    conv_true = [t for t, c in zip(true_elos, converged) if c]
    conv_rounds = [r for r, c in zip(rounds, converged) if c]
    not_conv_true = [t for t, c in zip(true_elos, converged) if not c]
    not_conv_rounds = [r for r, c in zip(rounds, converged) if not c]
    
    ax.scatter(conv_true, conv_rounds, alpha=0.6, label='Converged', color='green')
    ax.scatter(not_conv_true, not_conv_rounds, alpha=0.6, marker='x', 
               label='Did not converge', color='red')
    
    ax.set_xlabel('True ELO', fontsize=12)
    ax.set_ylabel('Rounds to Convergence', fontsize=12)
    ax.set_title('Convergence Speed by Starting ELO', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence by ELO to {output_path}")
    
    plt.show()
    return fig


def generate_all_paper_figures(
    config,
    simulation_results: Dict,
    output_dir: str = 'figures/'
):
    """
    Batch generate all figures for paper.
    
    Args:
        config: AssessmentConfig instance
        simulation_results: Dict with simulation data
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Probability model comparison
    default_model = config.get_probability_model()
    asymmetric_model = ProbabilityModel(steepness_upper=0.8, steepness_lower=1.2, offset=0)
    
    models = {
        'Default (symmetric)': default_model,
        'Asymmetric (realistic)': asymmetric_model,
    }
    
    plot_probability_curves_comparison(
        models,
        output_path=os.path.join(output_dir, 'probability_curves.png')
    )
    
    # 2. Convergence distribution if we have results
    if 'rounds_by_model' in simulation_results:
        plot_convergence_distribution(
            simulation_results['rounds_by_model'],
            output_path=os.path.join(output_dir, 'convergence_distribution.png')
        )
    
    print(f"All figures saved to {output_dir}")


if __name__ == '__main__':
    # Demo with default model
    print("Visualization Module Demo")
    print("=" * 50)
    
    default_model = ProbabilityModel()
    asymmetric_model = ProbabilityModel(steepness_upper=0.8, steepness_lower=1.2, offset=0)
    
    models = {
        'Default (symmetric)': default_model,
        'Asymmetric (harder questions harder)': asymmetric_model,
    }
    
    print("\nProbability curves for different models:")
    for diff in [-1000, -500, 0, 500, 1000]:
        print(f"  ELO diff {diff:+5d}: ", end='')
        for name, model in models.items():
            print(f"{name}={model.get_probability(diff):.3f} ", end='')
        print()
