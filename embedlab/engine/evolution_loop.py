"""Main evolution loop orchestrating Challenger-Solver co-evolution."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging

from ..core.llm_client import LLMClient, LLMConfig
from ..core.embedding_client import EmbeddingClient, EmbeddingConfig
from ..config.evolution_config import EvolutionConfig, EvolutionState, InstructionGene
from ..config.task_goals import TaskType
from ..generation.challenger import ChallengerAgent
from ..generation.dataset_builder import DatasetBuilder
from ..evolution.solver import SolverAgent
from ..evolution.instruction_optimizer import InstructionOptimizer
from ..hierarchy import load_hierarchy
from .reward_calculator import RewardCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionEngine:
    """
    Main orchestrator for R-Zero style self-evolution.
    Manages the co-evolution of Challenger and Solver agents.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        hierarchy_path: str,
        output_dir: str,
        regenerate_dataset: bool = False
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.regenerate_dataset = regenerate_dataset

        # Load hierarchy
        self.hierarchy = load_hierarchy(hierarchy_path)

        # Initialize clients
        self.llm = LLMClient(LLMConfig(base_url=config.llm_endpoint))
        self.embedder = EmbeddingClient(EmbeddingConfig(base_url=config.embedding_endpoint))

        # Initialize agents
        self.challenger = ChallengerAgent(self.llm, self.embedder, config, self.hierarchy)
        self.solver = SolverAgent(self.llm, self.embedder, config, self.hierarchy)
        self.optimizer = InstructionOptimizer(self.llm, config, self.hierarchy)
        self.reward_calc = RewardCalculator(config)

        # Initialize dataset builder
        self.dataset_builder = DatasetBuilder(self.llm, config, self.hierarchy)

        # Evolution state
        self.state = EvolutionState()

        # Datasets
        self.train_data = []
        self.test_data = []
        self.validation_set = []  # Fixed validation set for consistent tracking
        self.fitness_history = []  # Track improvement over time

    def run(self) -> Dict:
        """
        Run the complete evolution process.

        Returns:
            Final results and best instruction
        """
        logger.info("Starting R-Zero evolution process")

        # Phase 1: Bootstrap with seed data
        logger.info("Phase 1: Generating seed dataset")
        self._bootstrap_dataset()

        # Phase 2: Initialize instruction population
        logger.info("Phase 2: Initializing instruction population")
        self.state.population = self.optimizer.initialize_population()

        # Phase 3: Evolution loop
        logger.info(f"Phase 3: Starting evolution for {self.config.generations} generations")

        for generation in range(self.config.generations):
            self.state.generation = generation
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation {generation + 1}/{self.config.generations}")

            # Show progress bar for long runs
            if self.fitness_history:
                progress = self._format_progress_bar(generation + 1, self.config.generations)
                logger.info(f"Progress: {progress}")

            logger.info(f"{'='*60}")

            # Step 1: Challenger generates new difficult queries
            logger.info("Step 1: Challenger generating queries...")
            new_queries = self._challenger_phase()

            # Step 2: Solver attempts routing with each instruction
            logger.info("Step 2: Solver testing instructions...")
            fitness_scores = self._solver_phase(new_queries)

            # Step 3: Evolve instructions based on performance
            logger.info("Step 3: Evolving instructions...")
            self.state.population = self._evolution_phase(fitness_scores)

            # Step 4: Update difficulty and record progress
            self._update_state()

            # Show generation summary
            if self.fitness_history:
                self._show_generation_summary()

            # Step 5: Checkpoint if needed
            if (generation + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Step 6: Check for early stopping
            if self._should_stop_early():
                logger.info("Early stopping criteria met")
                break

        # Phase 4: Final evaluation
        logger.info("\nPhase 4: Final evaluation")
        final_results = self._final_evaluation()

        # Save final results
        self._save_results(final_results)

        return final_results

    def _bootstrap_dataset(self):
        """Generate initial seed dataset from hierarchy or load existing."""
        train_path = self.output_dir / "train_data.csv"
        test_path = self.output_dir / "test_data.csv"
        metadata_path = self.output_dir / "dataset_metadata.json"

        # Check if we should use existing datasets
        if not self.regenerate_dataset and train_path.exists() and test_path.exists():
            logger.info("Found existing datasets, loading them...")

            # Check metadata to ensure compatibility
            metadata_matches = self._check_dataset_metadata(metadata_path)

            if metadata_matches:
                # Load existing datasets
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                self.train_data = list(zip(train_df["text"], train_df["label_path"]))
                self.test_data = list(zip(test_df["text"], test_df["label_path"]))

                logger.info(f"Loaded {len(self.train_data)} training examples")
                logger.info(f"Loaded {len(self.test_data)} test examples")
                return
            else:
                logger.info("Dataset metadata doesn't match current configuration, regenerating...")

        # Generate new dataset
        logger.info("Generating new dataset from scratch...")

        # Generate seed examples
        seed_df = self.dataset_builder.generate_seed_dataset()

        # Augment for more variety
        augmented_df = self.dataset_builder.augment_dataset(
            seed_df,
            factor=self.config.augmentation_factor
        )

        # Add boundary examples
        boundary_df = self.dataset_builder.generate_cross_boundary_examples(
            num_examples=20
        )

        # Combine all
        full_df = pd.concat([augmented_df, boundary_df], ignore_index=True)

        # Balance dataset
        balanced_df = self.dataset_builder.balance_dataset(full_df)

        # Split train/test/validation
        test_size = max(
            self.config.min_test_size,
            int(len(balanced_df) * self.config.test_split)
        )
        val_size = min(30, len(balanced_df) // 10)  # Small fixed validation set

        test_df = balanced_df.sample(n=test_size, random_state=42)
        remaining_df = balanced_df.drop(test_df.index)
        val_df = remaining_df.sample(n=val_size, random_state=42)
        train_df = remaining_df.drop(val_df.index)

        self.train_data = list(zip(train_df["text"], train_df["label_path"]))
        self.test_data = list(zip(test_df["text"], test_df["label_path"]))
        self.validation_set = list(zip(val_df["text"], val_df["label_path"]))

        # Save datasets
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Save metadata
        self._save_dataset_metadata(metadata_path)

        logger.info(f"Generated {len(self.train_data)} training examples")
        logger.info(f"Generated {len(self.test_data)} test examples")
        logger.info(f"Generated {len(self.validation_set)} validation examples")

    def _challenger_phase(self) -> List[Tuple[str, str]]:
        """Challenger generates new difficult queries."""
        # Get solver performance for difficulty adjustment
        solver_perf = {
            "accuracy": self.state.solver.current_accuracy
        } if self.state.solver.attempts > 0 else None

        # Generate challenging queries
        challenge_queries = self.challenger.generate_batch(
            batch_size=self.config.dataset_size_per_gen,
            solver_performance=solver_perf
        )

        # Convert to training format
        new_data = [
            (q.text, q.target_path)
            for q in challenge_queries
        ]

        # Add to training data
        self.train_data.extend(new_data)

        # Keep training data size manageable
        max_size = self.config.dataset_size_per_gen * 10
        if len(self.train_data) > max_size:
            self.train_data = self.train_data[-max_size:]

        logger.info(f"Generated {len(new_data)} new challenging queries")
        logger.info(f"Current difficulty: {self.challenger.state.current_difficulty:.2f}")

        return new_data

    def _solver_phase(
        self,
        new_queries: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Solver tests each instruction variant."""
        fitness_scores = {}
        validation_scores = {}

        # Combine new queries with some existing training data
        test_queries = new_queries[:self.config.dataset_size_per_gen // 2]
        if len(self.train_data) > 20:
            test_queries.extend(
                self.train_data[-20:]  # Recent training examples
            )

        for gene in self.state.population:
            # Reset solver state for fair comparison
            self.solver.reset_state()

            # Test routing with this instruction
            results = self.solver.route_queries(
                test_queries,
                gene,
                use_embeddings=True
            )

            # Calculate fitness
            accuracy = sum(r.success for r in results) / len(results)
            avg_confidence = sum(r.confidence for r in results) / len(results)

            # Fitness combines accuracy and confidence
            fitness = accuracy * 0.8 + avg_confidence * 0.2

            # Also test on validation set for consistent tracking
            if self.validation_set:
                val_results = self.solver.route_queries(
                    self.validation_set[:20],  # Use subset for speed
                    gene,
                    use_embeddings=True
                )
                val_accuracy = sum(r.success for r in val_results) / len(val_results)
                validation_scores[gene.id] = val_accuracy

            fitness_scores[gene.id] = fitness
            gene.fitness = fitness
            gene.metrics = {
                "accuracy": accuracy,
                "confidence": avg_confidence,
                "val_accuracy": validation_scores.get(gene.id, 0),
                "tested_on": len(test_queries)
            }

        # Update best instruction
        best_gene = max(self.state.population, key=lambda x: x.fitness)
        if not self.state.best_instruction or best_gene.fitness > self.state.best_fitness:
            self.state.best_instruction = best_gene
            self.state.best_fitness = best_gene.fitness

        # Track validation performance for progress monitoring
        if validation_scores:
            best_val_score = max(validation_scores.values())
            self.fitness_history.append({
                "generation": self.state.generation,
                "best_fitness": self.state.best_fitness,
                "best_val_accuracy": best_val_score,
                "mean_fitness": sum(fitness_scores.values()) / len(fitness_scores)
            })

            # Show improvement trend
            if len(self.fitness_history) > 1:
                prev_val = self.fitness_history[-2]["best_val_accuracy"]
                improvement = best_val_score - prev_val
                trend = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                logger.info(f"Validation accuracy: {best_val_score:.1%} {trend} ({improvement:+.1%})")
            else:
                logger.info(f"Validation accuracy: {best_val_score:.1%}")

        logger.info(f"Evolution fitness: {self.state.best_fitness:.3f}")
        logger.info(f"Best instruction: {self.state.best_instruction.content[:100]}...")

        return fitness_scores

    def _evolution_phase(
        self,
        fitness_scores: Dict[str, float]
    ) -> List[InstructionGene]:
        """Evolve instruction population."""
        # Get failure analysis from solver
        failure_analysis = self.solver.analyze_failures()

        # Evolve to next generation
        new_population = self.optimizer.evolve_generation(
            self.state.population,
            fitness_scores
        )

        # Add targeted mutations based on failures
        if failure_analysis.get("top_failures"):
            best_gene = max(new_population, key=lambda x: x.fitness)
            targeted = self.optimizer.generate_targeted_mutation(
                best_gene,
                failure_analysis
            )
            # Replace worst performer with targeted mutation
            new_population[-1] = targeted

        # Analyze new population
        pop_analysis = self.optimizer.analyze_population(new_population)
        logger.info(f"Population diversity: {pop_analysis['diversity']:.2f}")
        logger.info(f"Mean fitness: {pop_analysis['mean_fitness']:.3f}")

        return new_population

    def _update_state(self):
        """Update evolution state and metrics."""
        # Calculate rewards
        solver_reward = self.solver.calculate_reward()
        challenger_reward = self.challenger.calculate_reward(
            {"accuracy": self.state.solver.current_accuracy}
        )

        # Update state
        self.state.challenger = self.challenger.state
        self.state.solver = self.solver.state

        # Record history
        self.state.history.append({
            "generation": self.state.generation,
            "best_fitness": self.state.best_fitness,
            "solver_accuracy": self.state.solver.current_accuracy,
            "challenger_difficulty": self.state.challenger.current_difficulty,
            "solver_reward": solver_reward,
            "challenger_reward": challenger_reward,
            "timestamp": datetime.now().isoformat()
        })

    def _should_stop_early(self) -> bool:
        """Check if evolution should stop early."""
        # Stop if reached success threshold
        if self.state.best_fitness >= self.config.success_threshold:
            return True

        # Stop if no improvement for many generations
        if len(self.state.history) > 20:
            recent_fitness = [h["best_fitness"] for h in self.state.history[-20:]]
            if max(recent_fitness) - min(recent_fitness) < 0.01:
                logger.info("Fitness plateau detected")
                return True

        return False

    def _save_checkpoint(self):
        """Save checkpoint of current state."""
        checkpoint = {
            "generation": self.state.generation,
            "best_instruction": {
                "content": self.state.best_instruction.content,
                "fitness": self.state.best_fitness
            } if self.state.best_instruction else None,
            "population": [
                {
                    "id": g.id,
                    "content": g.content,
                    "fitness": g.fitness,
                    "metrics": g.metrics
                }
                for g in self.state.population
            ],
            "history": self.state.history
        }

        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_gen_{self.state.generation}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _final_evaluation(self) -> Dict:
        """Perform final evaluation on test set."""
        if not self.state.best_instruction:
            return {"error": "No best instruction found"}

        # Test on held-out test data
        self.solver.reset_state()
        results = self.solver.route_queries(
            self.test_data,
            self.state.best_instruction,
            use_embeddings=True
        )

        # Calculate metrics
        accuracy = sum(r.success for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Confusion matrix
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))
        for r in results:
            confusion[r.true_path][r.predicted_path] += 1

        return {
            "best_instruction": self.state.best_instruction.content,
            "test_accuracy": accuracy,
            "test_confidence": avg_confidence,
            "total_generations": self.state.generation + 1,
            "evolution_history": self.state.history,
            "confusion_matrix": dict(confusion),
            "final_population": [
                {
                    "content": g.content,
                    "fitness": g.fitness,
                    "metrics": g.metrics
                }
                for g in sorted(self.state.population, key=lambda x: x.fitness, reverse=True)[:5]
            ]
        }

    def _save_results(self, results: Dict):
        """Save final results to file."""
        results_path = self.output_dir / "evolution_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save best instruction separately
        if "best_instruction" in results:
            instruction_path = self.output_dir / "best_instruction.txt"
            with open(instruction_path, "w") as f:
                f.write(results["best_instruction"])

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Best instruction: {results.get('best_instruction', 'N/A')}")
        logger.info(f"Test accuracy: {results.get('test_accuracy', 0):.3f}")

    def _check_dataset_metadata(self, metadata_path: Path) -> bool:
        """Check if existing dataset metadata matches current configuration."""
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check if task goal matches
            if self.config.task_goal:
                if metadata.get('task_goal_name') != self.config.task_goal_name:
                    logger.info(f"Task goal mismatch: {metadata.get('task_goal_name')} vs {self.config.task_goal_name}")
                    return False

                # Check task goal objective if available
                if metadata.get('task_goal_objective') != self.config.task_goal.objective:
                    logger.info("Task goal objective has changed")
                    return False

            # Check hierarchy hash (simple check based on leaf count)
            leaf_count = len([n for n in self.hierarchy.iter_nodes() if n.is_leaf()])
            if metadata.get('hierarchy_leaf_count') != leaf_count:
                logger.info(f"Hierarchy structure has changed")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error reading dataset metadata: {e}")
            return False

    def _save_dataset_metadata(self, metadata_path: Path):
        """Save metadata about the generated dataset."""
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'task_goal_name': self.config.task_goal_name if self.config.task_goal else None,
            'task_goal_objective': self.config.task_goal.objective if self.config.task_goal else None,
            'task_goal_type': self.config.task_goal.type.value if self.config.task_goal else None,
            'hierarchy_leaf_count': len([n for n in self.hierarchy.iter_nodes() if n.is_leaf()]),
            'seed_examples_per_node': self.config.seed_examples_per_node,
            'augmentation_factor': self.config.augmentation_factor,
            'train_size': len(self.train_data),
            'test_size': len(self.test_data)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset metadata saved to {metadata_path}")

    def _format_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Format a simple progress bar."""
        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {current}/{total} ({percent:.0%})"

    def _show_generation_summary(self):
        """Show a summary of the current generation's performance."""
        if not self.fitness_history:
            return

        current = self.fitness_history[-1]

        # Build summary message
        summary_parts = []

        # Show validation trend if available
        if current.get("best_val_accuracy", 0) > 0:
            val_acc = current["best_val_accuracy"]
            summary_parts.append(f"Val Acc: {val_acc:.1%}")

            # Show trend over last 5 generations
            if len(self.fitness_history) >= 5:
                five_gen_ago = self.fitness_history[-5]["best_val_accuracy"]
                trend = val_acc - five_gen_ago
                if trend > 0:
                    summary_parts.append(f"↑{trend:.1%} over 5 gen")
                elif trend < 0:
                    summary_parts.append(f"↓{abs(trend):.1%} over 5 gen")

        # Show mean fitness
        if current.get("mean_fitness"):
            summary_parts.append(f"Mean Fit: {current['mean_fitness']:.3f}")

        # Show challenger difficulty
        if hasattr(self.challenger, 'state') and hasattr(self.challenger.state, 'current_difficulty'):
            summary_parts.append(f"Difficulty: {self.challenger.state.current_difficulty:.1%}")

        if summary_parts:
            logger.info(f"Summary: {' | '.join(summary_parts)}")
