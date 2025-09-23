"""Main R-Zero implementation following the paper's co-evolution algorithm"""
import os
import sys
import json
import torch
import gc
import warnings
from datetime import datetime
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.cli_formatter import CLIFormatter
from utils.dataset_filter import DatasetFilter
from agents import ChallengerAgent, SolverAgent

# Disable multiprocessing to avoid Windows issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Globally suppress the padding warnings that persist despite correct configuration
# These warnings come from transformers library internals and we've already set padding_side="left"
warnings.filterwarnings("ignore", message=".*decoder-only.*right-padding.*")
warnings.filterwarnings("ignore", message=".*right-padding.*")
warnings.filterwarnings("ignore", message=".*padding_side.*")


def clear_memory():
    """Clear GPU memory between iterations"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


class RZero:
    """Full R-Zero implementation with Challenger-Solver co-evolution

    Following the paper:
    - Both Challenger and Solver start from same base model (Gemma-3-1B)
    - Challenger trained to generate problems at optimal difficulty
    - Solver trained on filtered curriculum with pseudo-labels
    - Co-evolution through alternating GRPO training

    IMPORTANT: Each new RZero().run() call starts from fresh base models.
    Checkpoints saved during a run are used only for memory management
    within that specific run, not for resuming across different runs.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        output_dir: str = "outputs",
        use_gsm8k_bootstrap: bool = True,
        memory_mode: str = "auto"
    ):
        """Initialize R-Zero system

        Args:
            run_id: Unique identifier for this run
            output_dir: Base output directory
            use_gsm8k_bootstrap: Whether to bootstrap with GSM8K examples
            memory_mode: Memory management mode ("auto", "concurrent", "sequential")
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"{output_dir}/{self.run_id}"
        self.use_gsm8k_bootstrap = use_gsm8k_bootstrap
        self.memory_mode = memory_mode
        self.concurrent_loading_possible = False

        # Create output directories
        os.makedirs(f"{self.run_dir}/challenger", exist_ok=True)
        os.makedirs(f"{self.run_dir}/solver", exist_ok=True)
        os.makedirs(f"{self.run_dir}/samples", exist_ok=True)
        os.makedirs(f"{self.run_dir}/metrics", exist_ok=True)

        # Initialize dataset filter
        self.dataset_filter = DatasetFilter(delta=0.25)

        # Track evolution metrics
        self.evolution_history = []

        CLIFormatter.print_header("R-Zero: Self-Evolving Reasoning from Zero Data")
        CLIFormatter.print_status("Run ID", self.run_id)
        CLIFormatter.print_status("Output Directory", self.run_dir)

    def bootstrap_with_gsm8k(self, n_examples: int = 30) -> List[Dict]:
        """Bootstrap with GSM8K examples for initial training

        Args:
            n_examples: Number of GSM8K examples to use

        Returns:
            List of bootstrap problems
        """
        CLIFormatter.print_subheader("Bootstrapping with GSM8K")
        gsm8k = load_dataset("openai/gsm8k", "main", split=f"train[:{n_examples}]")

        bootstrap_data = []
        for example in gsm8k:
            answer = example["answer"].split("####")[-1].strip() if "####" in example["answer"] else example["answer"]
            bootstrap_data.append({
                "question": example["question"],
                "answer": answer
            })

        CLIFormatter.print_success(f"Loaded {len(bootstrap_data)} GSM8K examples")
        return bootstrap_data

    def check_concurrent_loading_feasibility(self) -> bool:
        """Check if both models can fit in VRAM simultaneously

        Returns:
            True if concurrent loading is possible, False otherwise
        """
        if not torch.cuda.is_available():
            return False

        # Estimate memory requirements (in GB)
        solver_memory = 4.5  # Gemma-3-1B with LoRA
        challenger_memory = 4.5  # Same model architecture
        overhead = 2.0  # Buffer for gradients, activations, etc.

        total_required = solver_memory + challenger_memory + overhead
        available_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Leave some buffer for safety
        return available_vram >= total_required * 1.1

    def run(
        self,
        num_iterations: int = 3,
        n_candidate_problems: int = 200,  # Reduced from 8000 for faster training
        challenger_steps: int = 5,
        solver_steps: int = 15,
        m_solver_samples: int = 3,  # Reduced from 10 for faster evaluation
        progressive_scaling: bool = True,  # Enable progressive scaling
        early_stopping: bool = True  # Enable early stopping for filtering
    ):
        """Run full R-Zero co-evolution

        IMPORTANT: Every call to run() starts from fresh base models (Gemma-3-1B).
        Checkpoints are used only for memory efficiency WITHIN this run.
        To resume from a previous run, that functionality will be added separately.

        Args:
            num_iterations: Number of co-evolution iterations
            n_candidate_problems: Number of candidate problems per iteration
            challenger_steps: GRPO steps for Challenger training
            solver_steps: GRPO steps for Solver training
            m_solver_samples: Number of solver samples for self-consistency
            progressive_scaling: Whether to progressively scale problem count
            early_stopping: Whether to use early stopping in filtering
        """
        CLIFormatter.print_header("Starting R-Zero Co-Evolution")
        CLIFormatter.print_status("Iterations", str(num_iterations))
        CLIFormatter.print_status("Initial candidate problems", str(n_candidate_problems))
        CLIFormatter.print_status("Challenger GRPO steps", str(challenger_steps))
        CLIFormatter.print_status("Solver GRPO steps", str(solver_steps))
        CLIFormatter.print_status("Solver samples (m)", str(m_solver_samples))
        CLIFormatter.print_status("Progressive scaling", "Enabled" if progressive_scaling else "Disabled")
        CLIFormatter.print_status("Early stopping", "Enabled" if early_stopping else "Disabled")

        # Check CUDA availability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            CLIFormatter.print_success(f"CUDA Available: {device_name} ({memory_gb:.1f} GB)")
        else:
            CLIFormatter.print_warning("CUDA not available - training will be slow")

        # Determine memory management strategy
        if self.memory_mode == "auto":
            self.concurrent_loading_possible = self.check_concurrent_loading_feasibility()
            actual_mode = "concurrent" if self.concurrent_loading_possible else "sequential"
            CLIFormatter.print_status("Memory Mode", f"Auto-selected: {actual_mode}")
        elif self.memory_mode == "concurrent":
            self.concurrent_loading_possible = True
            CLIFormatter.print_status("Memory Mode", "Concurrent (both models in VRAM)")
        else:  # sequential
            self.concurrent_loading_possible = False
            CLIFormatter.print_status("Memory Mode", "Sequential (swap models as needed)")

        # Initialize models from same base - ALWAYS fresh, never from old checkpoints
        CLIFormatter.print_subheader("Initializing Fresh Models from Base")
        CLIFormatter.print_info("Starting from raw Gemma-3-1B base model (no checkpoint loading)")

        # Pass persistent flag based on memory mode
        persistent = self.concurrent_loading_possible
        challenger = ChallengerAgent(persistent=persistent)  # Always loads fresh base model
        solver = SolverAgent(persistent=persistent)  # Always loads fresh base model

        # Bootstrap Solver if requested
        if self.use_gsm8k_bootstrap:
            bootstrap_data = self.bootstrap_with_gsm8k(30)

            CLIFormatter.print_subheader("Pre-training Solver on GSM8K")
            solver.train_with_grpo(bootstrap_data, max_steps=10)
            solver_checkpoint = solver.save_checkpoint(0, self.run_id)

            # Only reload if in sequential mode for memory efficiency
            if not self.concurrent_loading_possible:
                solver.cleanup()
                clear_memory()
                solver = SolverAgent(checkpoint_path=solver_checkpoint, persistent=persistent)

        # Track metrics for progressive scaling
        previous_accuracy = 0.0
        current_n_problems = min(100, n_candidate_problems) if progressive_scaling else n_candidate_problems
        current_m_samples = m_solver_samples

        # Main co-evolution loop
        for iteration in range(1, num_iterations + 1):
            CLIFormatter.print_header(f"Iteration {iteration}/{num_iterations}")

            # Apply progressive scaling based on previous accuracy
            if progressive_scaling and iteration > 1:
                if previous_accuracy > 0.5 and current_n_problems < 2000:
                    # Good performance - scale up
                    current_n_problems = min(int(current_n_problems * 1.5), 2000)
                    current_m_samples = min(current_m_samples + 1, 10)
                    CLIFormatter.print_success(f"Scaling up: {current_n_problems} problems, {current_m_samples} samples")
                elif previous_accuracy < 0.3 and current_n_problems > 50:
                    # Poor performance - scale down
                    current_n_problems = max(int(current_n_problems * 0.5), 50)
                    current_m_samples = max(current_m_samples - 1, 2)
                    CLIFormatter.print_warning(f"Scaling down: {current_n_problems} problems, {current_m_samples} samples")

            iteration_metrics = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "n_problems_target": current_n_problems,
                "m_samples": current_m_samples
            }

            # Phase 1: Challenger generates candidate problems
            CLIFormatter.print_subheader("Phase 1: Challenger Generates Problems")
            CLIFormatter.print_info(f"Generating {current_n_problems} candidate problems...")

            candidate_problems = challenger.generate_candidate_problems(
                num_problems=current_n_problems,
                temperature=0.8
            )

            CLIFormatter.print_success(f"Generated {len(candidate_problems)} candidate problems")
            iteration_metrics["n_candidate_problems"] = len(candidate_problems)

            # Show sample problem
            if candidate_problems:
                sample = candidate_problems[0]
                CLIFormatter.print_info("Sample problem:")
                CLIFormatter.print_status("Question", sample["question"][:100] + "..." if len(sample["question"]) > 100 else sample["question"])
                CLIFormatter.print_status("Answer", sample["answer"])

            # Phase 2: Filter problems based on Solver's empirical accuracy
            CLIFormatter.print_subheader("Phase 2: Dataset Filtering")

            filtered_problems, filter_stats = self.dataset_filter.filter_dataset(
                candidate_problems,
                solver,
                m_samples=current_m_samples,
                early_stopping=early_stopping
            )

            iteration_metrics["filter_stats"] = filter_stats

            # Check if we have enough problems
            if len(filtered_problems) < 10:
                CLIFormatter.print_warning(f"Only {len(filtered_problems)} problems passed filtering - generating more...")

                # Generate additional problems with adjusted temperature
                additional_problems = challenger.generate_candidate_problems(
                    num_problems=100,
                    temperature=0.9  # Higher temperature for diversity
                )

                additional_filtered, _ = self.dataset_filter.filter_dataset(
                    additional_problems,
                    solver,
                    m_samples=current_m_samples,
                    early_stopping=early_stopping
                )

                filtered_problems.extend(additional_filtered)

            CLIFormatter.print_success(f"Final dataset size: {len(filtered_problems)} problems")

            # Save sample problems
            with open(f"{self.run_dir}/samples/iteration_{iteration}.json", "w") as f:
                json.dump({
                    "iteration": iteration,
                    "total_generated": len(candidate_problems),
                    "filtered_count": len(filtered_problems),
                    "samples": filtered_problems[:5] if filtered_problems else []
                }, f, indent=2)

            # Phase 3: Train Solver on filtered dataset
            if filtered_problems:
                CLIFormatter.print_subheader("Phase 3: Solver Training")
                solver.train_with_grpo(filtered_problems, max_steps=solver_steps)

                # Save solver checkpoint
                solver_checkpoint = solver.save_checkpoint(iteration, self.run_id)

                # Evaluate solver on test set
                test_problems = filtered_problems[:10]  # Use subset for testing
                test_results = solver.solve_problems(test_problems)
                solver_accuracy = test_results[0]["accuracy"] if test_results else 0.0
                previous_accuracy = solver_accuracy  # Store for progressive scaling

                iteration_metrics["solver_accuracy"] = solver_accuracy
                CLIFormatter.print_status("Solver test accuracy", f"{solver_accuracy:.2%}")
            else:
                CLIFormatter.print_warning("No problems passed filtering - skipping Solver training")
                solver_accuracy = 0.0
                iteration_metrics["solver_accuracy"] = solver_accuracy

            # Phase 4: Train Challenger with frozen Solver
            CLIFormatter.print_subheader("Phase 4: Challenger Training")

            # Freeze solver for Challenger training
            challenger.train_with_grpo(
                frozen_solver=solver,
                num_problems=min(50, len(filtered_problems)),  # Smaller batch for Challenger
                max_steps=challenger_steps,
                m_solver_samples=current_m_samples
            )

            # Save challenger checkpoint
            challenger_checkpoint = challenger.save_checkpoint(iteration, self.run_id)

            # Record iteration metrics
            self.evolution_history.append(iteration_metrics)

            # Save metrics
            with open(f"{self.run_dir}/metrics/evolution_history.json", "w") as f:
                json.dump(self.evolution_history, f, indent=2)

            # Clean up models for next iteration (only if in sequential mode)
            if not self.concurrent_loading_possible:
                CLIFormatter.print_info("Cleaning up models for next iteration...")
                solver.cleanup()
                challenger.cleanup()
                clear_memory()

                # Reload models from checkpoints for next iteration
                if iteration < num_iterations:
                    challenger = ChallengerAgent(checkpoint_path=challenger_checkpoint, persistent=persistent)
                    solver = SolverAgent(checkpoint_path=solver_checkpoint, persistent=persistent)
            else:
                CLIFormatter.print_info("Keeping models in VRAM (concurrent mode)")

            CLIFormatter.print_success(f"Iteration {iteration} complete!")
            print()

        # Final summary
        CLIFormatter.print_header("R-Zero Co-Evolution Complete!")
        self._print_summary()

    def _print_summary(self):
        """Print summary of co-evolution results"""
        CLIFormatter.print_subheader("Evolution Summary")

        if self.evolution_history:
            # Extract key metrics
            iterations = [m["iteration"] for m in self.evolution_history]
            accuracies = [m.get("solver_accuracy", 0) for m in self.evolution_history]

            # Print trajectory
            CLIFormatter.print_info("Solver accuracy trajectory:")
            for it, acc in zip(iterations, accuracies):
                bar = "█" * int(acc * 50)
                CLIFormatter.print_status(f"Iteration {it}", f"{bar} {acc:.2%}")

            # Best performance
            best_idx = accuracies.index(max(accuracies))
            CLIFormatter.print_success(f"Best accuracy: {accuracies[best_idx]:.2%} at iteration {iterations[best_idx]}")

            # Problem generation statistics
            total_generated = sum(m.get("n_candidate_problems", 0) for m in self.evolution_history)
            total_filtered = sum(m.get("filter_stats", {}).get("kept", 0) for m in self.evolution_history)
            CLIFormatter.print_status("Total problems generated", str(total_generated))
            CLIFormatter.print_status("Total problems kept after filtering", str(total_filtered))

        # Output locations
        CLIFormatter.print_subheader("Output Files")
        CLIFormatter.print_status("Run directory", self.run_dir)
        CLIFormatter.print_status("Challenger checkpoints", f"{self.run_dir}/challenger/")
        CLIFormatter.print_status("Solver checkpoints", f"{self.run_dir}/solver/")
        CLIFormatter.print_status("Sample problems", f"{self.run_dir}/samples/")
        CLIFormatter.print_status("Evolution metrics", f"{self.run_dir}/metrics/")


def main():
    """Main entry point for R-Zero

    IMPORTANT: Every execution starts from fresh base models (Gemma-3-1B).
    Previous run checkpoints are NOT loaded.
    """
    import argparse

    CLIFormatter.print_warning("=" * 60)
    CLIFormatter.print_warning("Starting NEW R-Zero run from FRESH BASE MODELS")
    CLIFormatter.print_warning("Previous checkpoints will NOT be loaded")
    CLIFormatter.print_warning("=" * 60)
    print()

    parser = argparse.ArgumentParser(description="R-Zero: Self-Evolving Reasoning from Zero Data")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of co-evolution iterations (default: 3)"
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=200,
        help="Initial number of candidate problems per iteration (default: 200)"
    )
    parser.add_argument(
        "--challenger-steps",
        type=int,
        default=5,
        help="GRPO training steps for Challenger (default: 5)"
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=15,
        help="GRPO training steps for Solver (default: 15)"
    )
    parser.add_argument(
        "--m-samples",
        type=int,
        default=3,
        help="Number of solver samples for self-consistency (default: 3)"
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip GSM8K bootstrapping"
    )
    parser.add_argument(
        "--no-progressive-scaling",
        action="store_true",
        help="Disable progressive scaling of problem count"
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping in dataset filtering"
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        choices=["auto", "concurrent", "sequential"],
        default="auto",
        help="Memory management mode: auto (detect), concurrent (both models in VRAM), sequential (swap models)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with minimal iterations"
    )

    args = parser.parse_args()

    # Test mode for quick validation
    if args.test:
        rzero = RZero(
            use_gsm8k_bootstrap=not args.no_bootstrap,
            memory_mode=args.memory_mode
        )
        rzero.run(
            num_iterations=2,
            n_candidate_problems=50,
            challenger_steps=2,
            solver_steps=3,
            m_solver_samples=2,
            progressive_scaling=not args.no_progressive_scaling,
            early_stopping=not args.no_early_stopping
        )
    else:
        rzero = RZero(
            use_gsm8k_bootstrap=not args.no_bootstrap,
            memory_mode=args.memory_mode
        )
        rzero.run(
            num_iterations=args.iterations,
            n_candidate_problems=args.n_problems,
            challenger_steps=args.challenger_steps,
            solver_steps=args.solver_steps,
            m_solver_samples=args.m_samples,
            progressive_scaling=not args.no_progressive_scaling,
            early_stopping=not args.no_early_stopping
        )


if __name__ == "__main__":
    main()
