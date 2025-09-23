"""Dataset filtering utilities for R-Zero curriculum generation"""
import sys
import os
import time
import random
from typing import List, Dict, Tuple, Optional

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
        batch_size: int = 4,
        early_stopping: bool = True,
        sample_size: int = 50
    ) -> Tuple[List[Dict], Dict]:
        """Filter problems based on solver's empirical accuracy

        Args:
            problems: List of candidate problems from Challenger
            solver: Solver model to evaluate problems
            m_samples: Number of solver samples per problem
            batch_size: Batch size for evaluation
            early_stopping: Whether to use early stopping based on sample
            sample_size: Size of sample for early stopping check

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
            "accuracy_distribution": [],
            "early_stopped": False,
            "problems_evaluated": 0
        }

        # Remove invalid problems first
        valid_problems = []
        for problem in problems:
            if problem.get("question") and problem.get("answer"):
                valid_problems.append(problem)
            else:
                statistics["invalid"] += 1

        if not valid_problems:
            CLIFormatter.print_warning("No valid problems to filter")
            return filtered_problems, statistics

        # Early stopping: evaluate a sample first
        if early_stopping and len(valid_problems) > sample_size:
            CLIFormatter.print_info(f"Early stopping check: evaluating {sample_size} sample problems...")
            sample_problems = random.sample(valid_problems, sample_size)

            # Use batch processing if available
            if hasattr(solver, 'solve_with_self_consistency_batch'):
                questions = [p["question"] for p in sample_problems]
                results = solver.solve_with_self_consistency_batch(
                    questions, m_samples=m_samples, batch_size=batch_size
                )
            else:
                results = []
                for problem in sample_problems:
                    result = solver.solve_with_self_consistency(
                        problem["question"], m_samples=m_samples
                    )
                    results.append(result)

            # Analyze sample results
            sample_kept = 0
            sample_too_easy = 0
            sample_too_hard = 0

            for (pseudo_label, empirical_acc, _) in results:
                statistics["accuracy_distribution"].append(empirical_acc)
                if empirical_acc < self.min_accuracy:
                    sample_too_hard += 1
                elif empirical_acc > self.max_accuracy:
                    sample_too_easy += 1
                else:
                    sample_kept += 1

            # Check if we should stop early
            sample_keep_rate = sample_kept / sample_size
            if sample_keep_rate < 0.2:  # Less than 20% useful
                CLIFormatter.print_warning(
                    f"Early stopping: Only {sample_keep_rate:.1%} of sample in target range. "
                    f"Skipping full evaluation."
                )
                statistics["early_stopped"] = True
                statistics["problems_evaluated"] = sample_size

                # Scale up statistics estimates
                scale_factor = len(valid_problems) / sample_size
                statistics["kept"] = int(sample_kept * scale_factor)
                statistics["too_easy"] = int(sample_too_easy * scale_factor)
                statistics["too_hard"] = int(sample_too_hard * scale_factor)

                # Still return the good problems from the sample
                for i, problem in enumerate(sample_problems):
                    pseudo_label, empirical_acc, _ = results[i]
                    if self.min_accuracy <= empirical_acc <= self.max_accuracy:
                        filtered_problem = problem.copy()
                        filtered_problem["pseudo_label"] = pseudo_label
                        filtered_problem["answer"] = pseudo_label  # IMPORTANT: Use pseudo_label as answer for training
                        filtered_problem["empirical_accuracy"] = empirical_acc
                        filtered_problems.append(filtered_problem)

                return filtered_problems, statistics
            else:
                CLIFormatter.print_success(
                    f"Sample shows {sample_keep_rate:.1%} in target range. Proceeding with full evaluation."
                )

        # Full evaluation with batch processing
        start_time = time.time()

        # Use batch processing if available
        if hasattr(solver, 'solve_with_self_consistency_batch'):
            CLIFormatter.print_info("Using batch processing for faster evaluation...")

            # Process in chunks for progress updates
            chunk_size = min(100, len(valid_problems))

            with SpinnerProgress(f"Evaluating {len(valid_problems)} problems") as spinner:
                for chunk_start in range(0, len(valid_problems), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(valid_problems))
                    chunk_problems = valid_problems[chunk_start:chunk_end]

                    spinner.update(f"Processing problems {chunk_start+1}-{chunk_end}/{len(valid_problems)}")

                    questions = [p["question"] for p in chunk_problems]
                    results = solver.solve_with_self_consistency_batch(
                        questions, m_samples=m_samples, batch_size=batch_size
                    )

                    # Process results
                    for problem, (pseudo_label, empirical_acc, solutions) in zip(chunk_problems, results):
                        statistics["accuracy_distribution"].append(empirical_acc)
                        statistics["problems_evaluated"] += 1

                        if empirical_acc < self.min_accuracy:
                            statistics["too_hard"] += 1
                        elif empirical_acc > self.max_accuracy:
                            statistics["too_easy"] += 1
                        else:
                            statistics["kept"] += 1
                            filtered_problem = problem.copy()
                            filtered_problem["pseudo_label"] = pseudo_label
                            filtered_problem["answer"] = pseudo_label  # IMPORTANT: Use pseudo_label as answer for training
                            filtered_problem["empirical_accuracy"] = empirical_acc
                            filtered_problems.append(filtered_problem)

                    # Estimate time remaining
                    elapsed = time.time() - start_time
                    problems_done = chunk_end
                    if problems_done > 0:
                        rate = problems_done / elapsed
                        remaining = (len(valid_problems) - problems_done) / rate
                        spinner.update(
                            f"Processing {chunk_end}/{len(valid_problems)} "
                            f"({rate:.1f} problems/s, ~{remaining:.0f}s remaining)"
                        )
        else:
            # Fallback to sequential processing
            with SpinnerProgress(f"Evaluating {len(valid_problems)} problems") as spinner:
                for idx, problem in enumerate(valid_problems):
                    spinner.update(f"Evaluating problem {idx+1}/{len(valid_problems)}")

                    pseudo_label, empirical_acc, solutions = solver.solve_with_self_consistency(
                        problem["question"], m_samples=m_samples
                    )

                    statistics["accuracy_distribution"].append(empirical_acc)
                    statistics["problems_evaluated"] += 1

                    if empirical_acc < self.min_accuracy:
                        statistics["too_hard"] += 1
                    elif empirical_acc > self.max_accuracy:
                        statistics["too_easy"] += 1
                    else:
                        statistics["kept"] += 1
                        filtered_problem = problem.copy()
                        filtered_problem["pseudo_label"] = pseudo_label
                        filtered_problem["answer"] = pseudo_label  # IMPORTANT: Use pseudo_label as answer for training
                        filtered_problem["empirical_accuracy"] = empirical_acc
                        filtered_problems.append(filtered_problem)

        # Calculate processing time
        total_time = time.time() - start_time
        if statistics["problems_evaluated"] > 0:
            CLIFormatter.print_success(
                f"Evaluated {statistics['problems_evaluated']} problems in {total_time:.1f}s "
                f"({statistics['problems_evaluated']/total_time:.1f} problems/s)"
            )

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