"""R-Zero Evolution Engine

Orchestrates the co-evolution between Challenger and Solver agents,
implementing the iterative training loop from the R-Zero paper.
"""

import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class RZeroEvolutionEngine:
    """Main engine for R-Zero co-evolution"""

    def __init__(
        self,
        challenger_agent,
        solver_agent,
        config,
        gsm8k_handler=None,
        output_dir: str = "outputs/rzero"
    ):
        """Initialize the evolution engine

        Args:
            challenger_agent: ChallengerAgent instance
            solver_agent: SolverAgent instance
            config: RZeroConfig with all settings
            gsm8k_handler: GSM8KHandler for evaluation
            output_dir: Directory for outputs and checkpoints
        """
        self.challenger = challenger_agent
        self.solver = solver_agent
        self.config = config
        self.gsm8k_handler = gsm8k_handler
        self.output_dir = output_dir

        # Evolution state
        self.current_iteration = 0
        self.evolution_history = []
        self.best_solver_accuracy = 0.0
        self.patience_counter = 0

        # Setup directories
        self._setup_directories()

        # Load any existing checkpoint
        self._load_latest_checkpoint()

    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)

    def run_evolution(self, num_iterations: Optional[int] = None):
        """Run the complete co-evolution loop

        Args:
            num_iterations: Number of iterations (uses config default if None)
        """
        num_iterations = num_iterations or self.config.evolution.num_iterations

        logger.info(f"Starting R-Zero evolution for {num_iterations} iterations")
        logger.info(f"Challenger: {self.config.challenger.model_name}")
        logger.info(f"Solver: {self.config.solver.model_name}")

        try:
            from ..utils.cli_formatter import CLIFormatter, Fore, Style
            use_formatter = True
        except ImportError:
            use_formatter = False

        for iteration in range(self.current_iteration, num_iterations):
            self.current_iteration = iteration

            if use_formatter:
                CLIFormatter.print_header(f"R-Zero Evolution - Iteration {iteration + 1}/{num_iterations}")

            iteration_start = time.time()
            iteration_metrics = {}

            # Step 1: Challenger generates difficult questions
            if use_formatter:
                CLIFormatter.print_subheader("Phase 1: Challenger Training")

            challenger_metrics = self._train_challenger()
            iteration_metrics["challenger"] = challenger_metrics

            # Step 2: Generate questions for Solver training
            if use_formatter:
                CLIFormatter.print_subheader("Phase 2: Question Generation")

            questions = self._generate_training_questions()
            iteration_metrics["num_questions_generated"] = len(questions)

            if len(questions) == 0:
                logger.warning("No questions generated, skipping iteration")
                continue

            # Step 3: Solver creates dataset with pseudo-labels
            if use_formatter:
                CLIFormatter.print_subheader("Phase 3: Dataset Construction")

            solver_dataset = self.solver.generate_dataset_with_voting(questions)
            iteration_metrics["num_training_examples"] = len(solver_dataset)

            if len(solver_dataset) == 0:
                logger.warning("No valid training examples after filtering, skipping iteration")
                continue

            # Log dataset statistics
            avg_confidence = np.mean([d.confidence for d in solver_dataset])
            iteration_metrics["avg_pseudo_label_confidence"] = avg_confidence
            logger.info(f"Created {len(solver_dataset)} training examples with avg confidence {avg_confidence:.3f}")

            # Step 4: Train Solver on the filtered dataset
            if use_formatter:
                CLIFormatter.print_subheader("Phase 4: Solver Training")

            solver_metrics = self._train_solver(solver_dataset)
            iteration_metrics["solver"] = solver_metrics

            # Step 5: Evaluate Solver on GSM8K (if available)
            if self.gsm8k_handler and iteration % 1 == 0:  # Evaluate every iteration
                if use_formatter:
                    CLIFormatter.print_subheader("Phase 5: GSM8K Evaluation")

                eval_metrics = self._evaluate_on_gsm8k()
                iteration_metrics["gsm8k_evaluation"] = eval_metrics

                # Check for improvement
                current_accuracy = eval_metrics["accuracy"]
                if current_accuracy > self.best_solver_accuracy:
                    improvement = current_accuracy - self.best_solver_accuracy
                    self.best_solver_accuracy = current_accuracy
                    self.patience_counter = 0
                    logger.info(f"New best accuracy: {current_accuracy:.3f} (+{improvement:.3f})")

                    # Save best checkpoint
                    self._save_best_checkpoint()
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement. Patience: {self.patience_counter}/{self.config.evolution.patience}")

            # Step 6: Save checkpoint
            if self.config.evolution.save_checkpoints:
                self._save_checkpoint(iteration_metrics)

            # Step 7: Log samples
            if self.config.evolution.log_samples:
                self._log_samples(questions[:self.config.evolution.samples_per_iteration])

            # Record iteration metrics
            iteration_time = time.time() - iteration_start
            iteration_metrics["iteration_time_seconds"] = iteration_time
            self.evolution_history.append(iteration_metrics)

            # Display iteration summary
            if use_formatter:
                self._display_iteration_summary(iteration_metrics)

            # Early stopping check
            if self._should_stop_early():
                logger.info("Early stopping triggered")
                break

            # Clear CUDA cache between iterations
            if torch.cuda.is_available() and iteration < num_iterations - 1:
                torch.cuda.empty_cache()

        # Final summary
        if use_formatter:
            self._display_final_summary()

        logger.info("R-Zero evolution complete")

    def _train_challenger(self) -> Dict[str, float]:
        """Train the Challenger agent"""
        logger.info("Training Challenger...")

        # Get current Solver model for uncertainty calculation
        solver_model = self.solver.get_model_for_challenger()

        # Train Challenger with GRPO
        metrics = self.challenger.train_with_grpo(
            solver_model=solver_model,
            num_steps=self.config.challenger.max_steps
        )

        return metrics

    def _generate_training_questions(self) -> List[str]:
        """Generate questions for Solver training"""
        logger.info(f"Generating {self.config.challenger.num_questions_per_iteration} questions...")

        # Get current Solver model
        solver_model = self.solver.get_model_for_challenger()

        # Generate questions
        questions_obj = self.challenger.generate_questions(
            num_questions=self.config.challenger.num_questions_per_iteration,
            solver_model=solver_model
        )

        # Extract question texts
        questions = [q.text for q in questions_obj]

        # Log statistics
        if questions_obj:
            avg_difficulty = np.mean([q.difficulty_estimate for q in questions_obj])
            avg_reward = np.mean([q.reward for q in questions_obj])
            logger.info(f"Generated {len(questions)} questions")
            logger.info(f"  Avg difficulty: {avg_difficulty:.3f}")
            logger.info(f"  Avg reward: {avg_reward:.3f}")

        return questions

    def _train_solver(self, dataset: List) -> Dict[str, float]:
        """Train the Solver agent"""
        logger.info(f"Training Solver on {len(dataset)} examples...")

        # Train Solver with GRPO
        metrics = self.solver.train_with_grpo(
            dataset=dataset,
            num_steps=self.config.solver.max_steps
        )

        return metrics

    def _evaluate_on_gsm8k(self) -> Dict[str, float]:
        """Evaluate Solver on GSM8K test set"""
        if not self.gsm8k_handler:
            return {"accuracy": 0.0}

        logger.info("Evaluating on GSM8K test set...")

        # Get test questions and answers
        test_data = self.gsm8k_handler.get_test_set(limit=200)  # Use subset for faster evaluation
        questions = [d["question"] for d in test_data]
        answers = [d["answer"] for d in test_data]

        # Evaluate Solver
        metrics = self.solver.evaluate_on_dataset(questions, answers)

        return metrics

    def _save_checkpoint(self, iteration_metrics: Dict[str, Any]):
        """Save checkpoint for current iteration"""
        checkpoint_dir = os.path.join(
            self.output_dir,
            "checkpoints",
            f"iteration_{self.current_iteration}"
        )

        # Save Challenger checkpoint
        challenger_dir = os.path.join(checkpoint_dir, "challenger")
        self.challenger.save_checkpoint(challenger_dir)

        # Save Solver checkpoint
        solver_dir = os.path.join(checkpoint_dir, "solver")
        self.solver.save_checkpoint(solver_dir)

        # Save evolution state
        state = {
            "current_iteration": self.current_iteration,
            "best_solver_accuracy": self.best_solver_accuracy,
            "patience_counter": self.patience_counter,
            "evolution_history": self.evolution_history,
            "iteration_metrics": iteration_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(checkpoint_dir, "evolution_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def _save_best_checkpoint(self):
        """Save the best Solver checkpoint"""
        best_dir = os.path.join(self.output_dir, "checkpoints", "best")
        self.solver.save_checkpoint(best_dir)
        logger.info(f"Best checkpoint saved to {best_dir}")

    def _load_latest_checkpoint(self):
        """Load the most recent checkpoint if it exists"""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            return

        # Find latest iteration checkpoint
        iterations = []
        for name in os.listdir(checkpoint_dir):
            if name.startswith("iteration_"):
                try:
                    iteration = int(name.split("_")[1])
                    iterations.append(iteration)
                except:
                    continue

        if not iterations:
            return

        latest_iteration = max(iterations)
        latest_dir = os.path.join(checkpoint_dir, f"iteration_{latest_iteration}")

        # Load evolution state
        state_file = os.path.join(latest_dir, "evolution_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)

            self.current_iteration = state["current_iteration"] + 1  # Start from next iteration
            self.best_solver_accuracy = state.get("best_solver_accuracy", 0.0)
            self.patience_counter = state.get("patience_counter", 0)
            self.evolution_history = state.get("evolution_history", [])

            logger.info(f"Resuming from iteration {self.current_iteration}")

            # Load model checkpoints
            try:
                self.challenger.load_checkpoint(os.path.join(latest_dir, "challenger"))
                self.solver.load_checkpoint(os.path.join(latest_dir, "solver"))
                logger.info("Model checkpoints loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model checkpoints: {e}")

    def _log_samples(self, questions: List[str]):
        """Log sample generated questions"""
        samples_file = os.path.join(
            self.output_dir,
            "samples",
            f"iteration_{self.current_iteration}_samples.txt"
        )

        with open(samples_file, "w", encoding="utf-8") as f:
            f.write(f"Iteration {self.current_iteration} - Sample Questions\n")
            f.write("=" * 80 + "\n\n")

            for i, question in enumerate(questions, 1):
                f.write(f"Question {i}:\n")
                f.write(question + "\n")
                f.write("-" * 40 + "\n\n")

        logger.info(f"Samples saved to {samples_file}")

    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered"""
        if not self.config.evolution.enable_early_stopping:
            return False

        if self.patience_counter >= self.config.evolution.patience:
            logger.info(f"Early stopping: no improvement for {self.patience_counter} iterations")
            return True

        # Check for minimum improvement
        if len(self.evolution_history) >= 2:
            recent_metrics = self.evolution_history[-2:]
            if "gsm8k_evaluation" in recent_metrics[0] and "gsm8k_evaluation" in recent_metrics[1]:
                prev_acc = recent_metrics[0]["gsm8k_evaluation"]["accuracy"]
                curr_acc = recent_metrics[1]["gsm8k_evaluation"]["accuracy"]
                improvement = curr_acc - prev_acc

                if improvement < self.config.evolution.min_improvement:
                    logger.info(f"Improvement {improvement:.4f} below threshold {self.config.evolution.min_improvement}")
                    return True

        return False

    def _display_iteration_summary(self, metrics: Dict[str, Any]):
        """Display summary of iteration metrics"""
        try:
            from ..utils.cli_formatter import CLIFormatter, Fore, Style
        except ImportError:
            return

        CLIFormatter.print_box_start("Iteration Summary", color=Fore.CYAN)

        # Display key metrics
        print(f"{Fore.WHITE}Questions Generated:{Style.RESET_ALL} {metrics.get('num_questions_generated', 0)}")
        print(f"{Fore.WHITE}Training Examples:{Style.RESET_ALL} {metrics.get('num_training_examples', 0)}")
        print(f"{Fore.WHITE}Avg Confidence:{Style.RESET_ALL} {metrics.get('avg_pseudo_label_confidence', 0):.3f}")

        if "gsm8k_evaluation" in metrics:
            eval_metrics = metrics["gsm8k_evaluation"]
            accuracy = eval_metrics["accuracy"]
            color = Fore.GREEN if accuracy > 0.5 else (Fore.YELLOW if accuracy > 0.3 else Fore.RED)
            print(f"{Fore.WHITE}GSM8K Accuracy:{Style.RESET_ALL} {color}{accuracy:.3f}{Style.RESET_ALL} ({eval_metrics['correct']}/{eval_metrics['total']})")

        print(f"{Fore.WHITE}Iteration Time:{Style.RESET_ALL} {metrics.get('iteration_time_seconds', 0):.1f}s")

        CLIFormatter.print_box_end()

    def _display_final_summary(self):
        """Display final evolution summary"""
        try:
            from ..utils.cli_formatter import CLIFormatter, Fore, Style
        except ImportError:
            return

        CLIFormatter.print_header("R-Zero Evolution Complete")

        print(f"{Fore.WHITE}Total Iterations:{Style.RESET_ALL} {len(self.evolution_history)}")
        print(f"{Fore.WHITE}Best GSM8K Accuracy:{Style.RESET_ALL} {Fore.GREEN}{self.best_solver_accuracy:.3f}{Style.RESET_ALL}")

        # Show accuracy progression
        if self.evolution_history:
            print(f"\n{Fore.CYAN}Accuracy Progression:{Style.RESET_ALL}")
            for i, metrics in enumerate(self.evolution_history):
                if "gsm8k_evaluation" in metrics:
                    acc = metrics["gsm8k_evaluation"]["accuracy"]
                    print(f"  Iteration {i+1}: {acc:.3f}")

        print(f"\n{Fore.GREEN}✓ Evolution complete!{Style.RESET_ALL}")
