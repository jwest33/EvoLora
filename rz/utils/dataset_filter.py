"""Dataset filtering utilities for R-Zero curriculum generation"""
import sys
import os
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cli_formatter import CLIFormatter, SpinnerProgress


class DatasetFilter:
    """Filters generated problems based on solver performance following R-Zero paper

    Implements the curriculum generation strategy:
    - Keeps problems where solver accuracy is in informative band (30-70%)
    - Filters out too easy (>70% accuracy) and too hard (<30% accuracy) problems
    - Serves as implicit quality control by removing ambiguous/ill-posed questions
    """

    def __init__(self, delta: float = 0.25):
        """Initialize dataset filter

        Args:
            delta: Filtering threshold (default 0.25 means keep if |p̂ - 0.5| ≤ 0.25)
                  This corresponds to empirical accuracy between 25% and 75%
        """
        self.delta = delta
        self.min_accuracy = 0.5 - delta  # 0.25
        self.max_accuracy = 0.5 + delta  # 0.75

    def filter_dataset(
        self,
        problems: List[Dict],
        solver,
        m_samples: int = 10,
        batch_size: int = 10
    ) -> Tuple[List[Dict], Dict]:
        """Filter problems based on solver's empirical accuracy

        Args:
            problems: List of candidate problems from Challenger
            solver: Solver model to evaluate problems
            m_samples: Number of solver samples per problem
            batch_size: Batch size for evaluation

        Returns:
            Tuple of (filtered_problems, statistics)
        """
        CLIFormatter.print_info(f"Filtering {len(problems)} candidate problems...")

        filtered_problems = []
        statistics = {
            "total": len(problems),
            "kept": 0,
            "too_easy": 0,
            "too_hard": 0,
            "invalid": 0,
            "accuracy_distribution": []
        }

        with SpinnerProgress(f"Evaluating {len(problems)} problems") as spinner:
            for idx, problem in enumerate(problems):
                spinner.update(f"Evaluating problem {idx+1}/{len(problems)}")

                # Skip invalid problems
                if not problem.get("question") or not problem.get("answer"):
                    statistics["invalid"] += 1
                    continue

                # Get solver's empirical accuracy using self-consistency
                pseudo_label, empirical_acc, solutions = solver.solve_with_self_consistency(
                    problem["question"],
                    m_samples=m_samples
                )

                # Record accuracy for statistics
                statistics["accuracy_distribution"].append(empirical_acc)

                # Apply filtering criteria
                if empirical_acc < self.min_accuracy:
                    # Too hard or ambiguous
                    statistics["too_hard"] += 1
                elif empirical_acc > self.max_accuracy:
                    # Too easy
                    statistics["too_easy"] += 1
                else:
                    # In the informative band - keep it
                    statistics["kept"] += 1
                    filtered_problem = problem.copy()
                    filtered_problem["pseudo_label"] = pseudo_label
                    filtered_problem["empirical_accuracy"] = empirical_acc
                    filtered_problems.append(filtered_problem)

        # Print filtering statistics
        CLIFormatter.print_subheader("Dataset Filtering Results")
        CLIFormatter.print_status("Total problems", str(statistics["total"]))
        CLIFormatter.print_status("Kept (informative)", f"{statistics['kept']} ({statistics['kept']/statistics['total']*100:.1f}%)")
        CLIFormatter.print_status("Too easy (>75% acc)", f"{statistics['too_easy']} ({statistics['too_easy']/statistics['total']*100:.1f}%)")
        CLIFormatter.print_status("Too hard (<25% acc)", f"{statistics['too_hard']} ({statistics['too_hard']/statistics['total']*100:.1f}%)")
        if statistics["invalid"] > 0:
            CLIFormatter.print_warning(f"Invalid problems: {statistics['invalid']}")

        # Print accuracy distribution if we have data
        if statistics["accuracy_distribution"]:
            avg_acc = sum(statistics["accuracy_distribution"]) / len(statistics["accuracy_distribution"])
            CLIFormatter.print_status("Average empirical accuracy", f"{avg_acc:.2%}")

        return filtered_problems, statistics

    def compute_pseudo_labels(
        self,
        problems: List[Dict],
        solver,
        m_samples: int = 10
    ) -> List[Dict]:
        """Compute pseudo-labels for problems using majority voting

        Args:
            problems: List of problems needing labels
            solver: Solver model for generating solutions
            m_samples: Number of samples for majority voting

        Returns:
            Problems with pseudo-labels added
        """
        CLIFormatter.print_info(f"Computing pseudo-labels for {len(problems)} problems...")

        labeled_problems = []

        with SpinnerProgress(f"Computing pseudo-labels") as spinner:
            for idx, problem in enumerate(problems):
                spinner.update(f"Processing problem {idx+1}/{len(problems)}")

                # Get pseudo-label via self-consistency
                pseudo_label, empirical_acc, _ = solver.solve_with_self_consistency(
                    problem["question"],
                    m_samples=m_samples
                )

                # Add pseudo-label to problem
                labeled_problem = problem.copy()
                labeled_problem["answer"] = pseudo_label  # Use pseudo-label as answer
                labeled_problem["empirical_accuracy"] = empirical_acc
                labeled_problem["is_pseudo_labeled"] = True
                labeled_problems.append(labeled_problem)

        return labeled_problems

    def analyze_difficulty_distribution(self, problems: List[Dict]) -> Dict:
        """Analyze the difficulty distribution of filtered problems

        Args:
            problems: List of filtered problems with empirical accuracies

        Returns:
            Dictionary with distribution statistics
        """
        if not problems:
            return {"error": "No problems to analyze"}

        accuracies = [p.get("empirical_accuracy", 0.5) for p in problems]

        # Compute statistics
        stats = {
            "count": len(problems),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "median_accuracy": sorted(accuracies)[len(accuracies) // 2],
            "in_target_range": sum(1 for a in accuracies if 0.4 <= a <= 0.6) / len(accuracies)
        }

        # Difficulty bins
        bins = {
            "very_hard": sum(1 for a in accuracies if a < 0.3),
            "hard": sum(1 for a in accuracies if 0.3 <= a < 0.4),
            "optimal": sum(1 for a in accuracies if 0.4 <= a <= 0.6),
            "easy": sum(1 for a in accuracies if 0.6 < a <= 0.7),
            "very_easy": sum(1 for a in accuracies if a > 0.7)
        }
        stats["difficulty_bins"] = bins

        return stats