"""
Monitoring utilities for evolution progress.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class EvolutionMonitor:
    """Monitor and visualize evolution progress."""

    def __init__(self, experiment_dir: str):
        """
        Initialize monitor.

        Args:
            experiment_dir: Directory containing experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.results_file = self.experiment_dir / 'evolution_results.json'
        self.history = []

    def load_results(self) -> Dict[str, Any]:
        """Load evolution results from file."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}

    def show_summary(self):
        """Display evolution summary."""
        results = self.load_results()

        if not results:
            print("No results found.")
            return

        print("\n" + "="*60)
        print("Evolution Summary")
        print("="*60)

        print(f"Total Generations: {results.get('total_generations', 0)}")
        print(f"Best Score: {results.get('best_score', 0):.4f}")
        print(f"Start Time: {results.get('timestamp', 'N/A')}")

        # Show progress table
        history = results.get('history', [])
        if history:
            print("\nProgress by Generation:")
            print("-"*50)
            print(f"{'Gen':>5} {'Score':>10} {'Loss':>10} {'Difficulty':>10} {'Time (s)':>10}")
            print("-"*50)

            for entry in history[-10:]:  # Show last 10
                metrics = entry['metrics']
                gen = entry['generation']
                score = metrics.get('eval_score', 0)
                loss = metrics.get('loss', 0)
                difficulty = metrics.get('difficulty', 0)
                iter_time = metrics.get('iteration_time', 0)

                print(f"{gen:5d} {score:10.3f} {loss:10.3f} {difficulty:10.2f} {iter_time:10.1f}")

    def live_monitor(self, refresh_interval: int = 5):
        """
        Live monitoring of ongoing evolution.

        Args:
            refresh_interval: Refresh interval in seconds
        """
        print(f"Monitoring {self.experiment_dir}")
        print(f"Refreshing every {refresh_interval} seconds...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                self._clear_screen()
                self.show_summary()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def generate_plots(self):
        """Generate visualization plots."""
        results = self.load_results()
        history = results.get('history', [])

        if not history:
            print("No history data to plot.")
            return

        # Extract data
        generations = [h['generation'] for h in history]
        scores = [h['metrics'].get('eval_score', 0) for h in history]
        losses = [h['metrics'].get('loss', 0) for h in history]
        difficulties = [h['metrics'].get('difficulty', 0) for h in history]
        task_qualities = [h['metrics'].get('task_quality', 0) for h in history]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('LoRA Evolution Progress', fontsize=16)

        # Score plot
        axes[0, 0].plot(generations, scores, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Evaluation Score')
        axes[0, 0].set_title('Model Performance')
        axes[0, 0].grid(True, alpha=0.3)

        # Loss plot
        axes[0, 1].plot(generations, losses, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # Difficulty plot
        axes[1, 0].plot(generations, difficulties, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Task Difficulty')
        axes[1, 0].set_title('Task Difficulty Progression')
        axes[1, 0].grid(True, alpha=0.3)

        # Task quality plot
        axes[1, 1].plot(generations, task_qualities, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Task Quality')
        axes[1, 1].set_title('Generated Task Quality')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_dir = self.experiment_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / 'evolution_progress.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Generate additional plots
        self._generate_correlation_plot(history, plot_dir)
        self._generate_convergence_plot(scores, plot_dir)

        print(f"Plots saved to {plot_dir}")

    def _generate_correlation_plot(self, history: List[Dict], plot_dir: Path):
        """Generate correlation plot between difficulty and performance."""
        difficulties = [h['metrics'].get('difficulty', 0) for h in history]
        scores = [h['metrics'].get('eval_score', 0) for h in history]

        plt.figure(figsize=(8, 6))
        plt.scatter(difficulties, scores, alpha=0.6, s=50)
        plt.xlabel('Task Difficulty')
        plt.ylabel('Evaluation Score')
        plt.title('Performance vs Difficulty')
        plt.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(difficulties, scores, 1)
        p = np.poly1d(z)
        plt.plot(difficulties, p(difficulties), "r--", alpha=0.8, label='Trend')
        plt.legend()

        plt.savefig(plot_dir / 'difficulty_correlation.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _generate_convergence_plot(self, scores: List[float], plot_dir: Path):
        """Generate convergence analysis plot."""
        if len(scores) < 2:
            return

        plt.figure(figsize=(8, 6))

        # Plot scores with rolling average
        window = min(5, len(scores) // 4)
        if window > 1:
            rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            x_rolling = range(window-1, len(scores))
            plt.plot(x_rolling, rolling_avg, 'r-', linewidth=2, label=f'{window}-Gen Average')

        plt.plot(scores, 'b-', alpha=0.5, label='Raw Score')
        plt.xlabel('Generation')
        plt.ylabel('Evaluation Score')
        plt.title('Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(plot_dir / 'convergence.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _clear_screen(self):
        """Clear console screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Find the best checkpoint directory.

        Returns:
            Path to best checkpoint or None
        """
        best_checkpoint = self.experiment_dir / 'best_checkpoint'
        if best_checkpoint.exists():
            return best_checkpoint

        # Find generation with highest score
        results = self.load_results()
        history = results.get('history', [])

        if not history:
            return None

        best_gen = max(history, key=lambda h: h['metrics'].get('eval_score', 0))
        gen_checkpoint = self.experiment_dir / f"generation_{best_gen['generation']:03d}"

        if gen_checkpoint.exists():
            return gen_checkpoint

        return None