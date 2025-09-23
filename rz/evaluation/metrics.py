"""Metrics and evaluation tools for R-Zero co-evolution tracking"""
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import os


class EvolutionMetrics:
    """Track and analyze R-Zero co-evolution metrics

    Following the paper, we track:
    - Question difficulty evolution
    - Pseudo-label accuracy degradation
    - Model performance trajectory
    - Diversity metrics
    """

    def __init__(self, run_dir: str):
        """Initialize metrics tracker

        Args:
            run_dir: Directory containing R-Zero run outputs
        """
        self.run_dir = run_dir
        self.metrics_dir = f"{run_dir}/metrics"
        self.evolution_data = []

        # Load existing metrics if available
        history_file = f"{self.metrics_dir}/evolution_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.evolution_data = json.load(f)

    def add_iteration_metrics(self, iteration_data: Dict):
        """Add metrics from a completed iteration

        Args:
            iteration_data: Dictionary containing iteration metrics
        """
        self.evolution_data.append(iteration_data)
        self.save_metrics()

    def save_metrics(self):
        """Save metrics to JSON file"""
        os.makedirs(self.metrics_dir, exist_ok=True)
        with open(f"{self.metrics_dir}/evolution_history.json", "w") as f:
            json.dump(self.evolution_data, f, indent=2)

    def analyze_trajectory(self) -> Dict:
        """Analyze the co-evolution trajectory

        Returns:
            Dictionary with trajectory analysis
        """
        if not self.evolution_data:
            return {"error": "No evolution data available"}

        analysis = {
            "num_iterations": len(self.evolution_data),
            "solver_accuracy_trajectory": [],
            "problem_generation_stats": [],
            "filtering_efficiency": []
        }

        for data in self.evolution_data:
            # Solver accuracy
            analysis["solver_accuracy_trajectory"].append(
                data.get("solver_accuracy", 0.0)
            )

            # Problem generation
            filter_stats = data.get("filter_stats", {})
            analysis["problem_generation_stats"].append({
                "iteration": data.get("iteration"),
                "total_generated": filter_stats.get("total", 0),
                "kept": filter_stats.get("kept", 0),
                "too_easy": filter_stats.get("too_easy", 0),
                "too_hard": filter_stats.get("too_hard", 0)
            })

            # Filtering efficiency
            total = filter_stats.get("total", 1)
            kept = filter_stats.get("kept", 0)
            efficiency = kept / total if total > 0 else 0
            analysis["filtering_efficiency"].append(efficiency)

        # Compute summary statistics
        accuracies = analysis["solver_accuracy_trajectory"]
        if accuracies:
            analysis["best_accuracy"] = max(accuracies)
            analysis["best_iteration"] = accuracies.index(max(accuracies)) + 1
            analysis["final_accuracy"] = accuracies[-1]
            analysis["average_accuracy"] = np.mean(accuracies)

            # Check for collapse
            if len(accuracies) >= 3:
                recent_trend = accuracies[-3:]
                if all(recent_trend[i] < recent_trend[i-1] for i in range(1, len(recent_trend))):
                    analysis["warning"] = "Performance degradation detected in last 3 iterations"

        return analysis

    def analyze_difficulty_evolution(self) -> Dict:
        """Analyze how problem difficulty evolves over iterations

        Returns:
            Dictionary with difficulty evolution metrics
        """
        difficulty_data = defaultdict(list)

        for iteration_data in self.evolution_data:
            filter_stats = iteration_data.get("filter_stats", {})
            acc_distribution = filter_stats.get("accuracy_distribution", [])

            if acc_distribution:
                difficulty_data["mean_difficulty"].append(np.mean(acc_distribution))
                difficulty_data["std_difficulty"].append(np.std(acc_distribution))
                difficulty_data["median_difficulty"].append(np.median(acc_distribution))

        return dict(difficulty_data)

    def detect_collapse(self, window_size: int = 3, threshold: float = 0.1) -> bool:
        """Detect if the system is experiencing model collapse

        Args:
            window_size: Number of recent iterations to check
            threshold: Minimum drop to consider as collapse

        Returns:
            True if collapse is detected
        """
        if len(self.evolution_data) < window_size:
            return False

        recent_accuracies = []
        for data in self.evolution_data[-window_size:]:
            recent_accuracies.append(data.get("solver_accuracy", 0.0))

        if not recent_accuracies:
            return False

        # Check for consistent degradation
        max_acc = max(recent_accuracies)
        min_acc = min(recent_accuracies)
        drop = max_acc - min_acc

        # Collapse if significant drop and downward trend
        is_downward = all(recent_accuracies[i] <= recent_accuracies[i-1]
                         for i in range(1, len(recent_accuracies)))

        return drop > threshold and is_downward

    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot co-evolution metrics

        Args:
            save_path: Path to save the plot
        """
        if not self.evolution_data:
            print("No evolution data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        iterations = [d.get("iteration", i+1) for i, d in enumerate(self.evolution_data)]

        # 1. Solver Accuracy
        accuracies = [d.get("solver_accuracy", 0) for d in self.evolution_data]
        axes[0, 0].plot(iterations, accuracies, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Solver Accuracy')
        axes[0, 0].set_title('Solver Performance Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # 2. Problem Generation Stats
        kept = []
        too_easy = []
        too_hard = []
        for d in self.evolution_data:
            stats = d.get("filter_stats", {})
            total = max(stats.get("total", 1), 1)
            kept.append(stats.get("kept", 0) / total)
            too_easy.append(stats.get("too_easy", 0) / total)
            too_hard.append(stats.get("too_hard", 0) / total)

        axes[0, 1].bar(iterations, kept, label='Kept', alpha=0.7, color='green')
        axes[0, 1].bar(iterations, too_easy, bottom=kept, label='Too Easy', alpha=0.7, color='yellow')
        axes[0, 1].bar(iterations, too_hard, bottom=[k+e for k, e in zip(kept, too_easy)],
                      label='Too Hard', alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Fraction of Problems')
        axes[0, 1].set_title('Problem Filtering Distribution')
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1])

        # 3. Filtering Efficiency
        efficiency = []
        for d in self.evolution_data:
            stats = d.get("filter_stats", {})
            total = max(stats.get("total", 1), 1)
            efficiency.append(stats.get("kept", 0) / total)

        axes[1, 0].plot(iterations, efficiency, 'g-s', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Filtering Efficiency')
        axes[1, 0].set_title('Fraction of Problems Kept After Filtering')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        # 4. Cumulative Problems
        cumulative_generated = []
        cumulative_kept = []
        total_gen = 0
        total_kept = 0
        for d in self.evolution_data:
            stats = d.get("filter_stats", {})
            total_gen += stats.get("total", 0)
            total_kept += stats.get("kept", 0)
            cumulative_generated.append(total_gen)
            cumulative_kept.append(total_kept)

        axes[1, 1].plot(iterations, cumulative_generated, 'b-', label='Generated', linewidth=2)
        axes[1, 1].plot(iterations, cumulative_kept, 'g-', label='Kept', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Cumulative Problems')
        axes[1, 1].set_title('Cumulative Problem Generation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self) -> str:
        """Generate a text report of co-evolution metrics

        Returns:
            Formatted report string
        """
        analysis = self.analyze_trajectory()
        difficulty = self.analyze_difficulty_evolution()

        report = []
        report.append("=" * 60)
        report.append("R-Zero Co-Evolution Report")
        report.append("=" * 60)

        # Summary
        report.append("\n[Summary]")
        report.append(f"Total Iterations: {analysis.get('num_iterations', 0)}")
        report.append(f"Best Accuracy: {analysis.get('best_accuracy', 0):.2%} (Iteration {analysis.get('best_iteration', 0)})")
        report.append(f"Final Accuracy: {analysis.get('final_accuracy', 0):.2%}")
        report.append(f"Average Accuracy: {analysis.get('average_accuracy', 0):.2%}")

        # Trajectory
        report.append("\n[Accuracy Trajectory]")
        for i, acc in enumerate(analysis.get('solver_accuracy_trajectory', [])):
            bar = "█" * int(acc * 30)
            report.append(f"Iteration {i+1}: {bar} {acc:.2%}")

        # Problem Generation
        report.append("\n[Problem Generation Statistics]")
        for stats in analysis.get('problem_generation_stats', []):
            report.append(f"\nIteration {stats['iteration']}:")
            report.append(f"  Generated: {stats['total_generated']}")
            report.append(f"  Kept: {stats['kept']} ({stats['kept']/max(stats['total_generated'], 1)*100:.1f}%)")
            report.append(f"  Too Easy: {stats['too_easy']}")
            report.append(f"  Too Hard: {stats['too_hard']}")

        # Warnings
        if analysis.get('warning'):
            report.append(f"\n⚠️ WARNING: {analysis['warning']}")

        if self.detect_collapse():
            report.append("\n⚠️ WARNING: Model collapse detected!")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def save_report(self, filename: Optional[str] = None):
        """Save evaluation report to file

        Args:
            filename: Output filename (default: metrics/report.txt)
        """
        if filename is None:
            filename = f"{self.metrics_dir}/report.txt"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        report = self.generate_report()

        with open(filename, 'w') as f:
            f.write(report)

        print(f"Report saved to {filename}")