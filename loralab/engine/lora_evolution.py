"""
Co-evolution engine for challenger-solver training loop.
Orchestrates the R-Zero inspired training process.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import torch

from loralab.core.llama_client import LlamaCppClient
from loralab.core.solver_client import SolverModel
from loralab.generation.task_challenger import TaskChallenger
from loralab.adaptation.lora_solver import LoRASolverTrainer

logger = logging.getLogger(__name__)


class LoRAEvolution:
    """Main co-evolution orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evolution engine.

        Args:
            config: Full configuration including challenger, solver, and training params
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self._setup_logging()

        # Initialize models
        logger.info("Initializing challenger and solver models")
        self.challenger_client = LlamaCppClient(config['challenger'])
        self.solver_model = SolverModel(config['solver'])

        # Initialize agents
        self.challenger = TaskChallenger(
            self.challenger_client,
            config['task']
        )
        self.trainer = LoRASolverTrainer(
            self.solver_model,
            config['training']
        )

        # Evolution parameters
        self.num_generations = config['evolution'].get('generations', 50)
        self.population_size = config['evolution'].get('population_size', 10)
        self.dataset_size = config['evolution'].get('dataset_size_per_gen', 100)
        self.bootstrap_size = config['evolution'].get('bootstrap_size', self.dataset_size)  # Default to dataset_size if not specified
        self.eval_ratio = config['evolution'].get('eval_ratio', 0.2)

        # Tracking
        self.generation = 0
        self.best_score = -float('inf')
        self.history = []

    def _setup_logging(self):
        """Configure logging for the evolution process."""
        log_file = self.output_dir / 'evolution.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def run(self):
        """Run the complete evolution process."""
        logger.info(f"Starting LoRA evolution for {self.num_generations} generations")
        logger.info(f"Output directory: {self.output_dir}")

        # Show task counts
        total_tasks = self.bootstrap_size + (self.num_generations * self.dataset_size)
        print(f"\n      === EVOLUTION OVERVIEW ===")
        print(f"      Bootstrap: {self.bootstrap_size} tasks (baseline evaluation only)")
        print(f"      Generations: {self.num_generations} x {self.dataset_size} tasks = {self.num_generations * self.dataset_size} tasks")
        print(f"      Total tasks to generate: {total_tasks}")
        print(f"      Estimated time: {total_tasks * 3 / 60:.1f} - {total_tasks * 5 / 60:.1f} minutes")
        print(f"      ========================\n")

        # Start challenger server
        self.challenger_client.start_server()

        try:
            # Bootstrap: Generate initial dataset
            logger.info("Bootstrapping with initial dataset")
            solver_performance = self._bootstrap()

            # Evolution loop
            for gen in range(self.num_generations):
                self.generation = gen + 1
                logger.info(f"\n{'='*50}")
                logger.info(f"Generation {self.generation}/{self.num_generations}")
                logger.info(f"Current solver performance: {solver_performance:.3f}")

                # Run one evolution iteration
                metrics = self._evolution_iteration(solver_performance)

                # Update solver performance
                solver_performance = metrics['eval_score']

                # Save checkpoint if improved
                if solver_performance > self.best_score:
                    self.best_score = solver_performance
                    self._save_best_checkpoint()

                # Save generation checkpoint
                self._save_generation_checkpoint(metrics)

                # Add to history
                self.history.append({
                    'generation': self.generation,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                })

                # Early stopping if perfect performance
                if solver_performance > 0.95:
                    logger.info("Achieved near-perfect performance. Stopping early.")
                    break

        finally:
            # Cleanup
            if self.using_direct:
                self.challenger_client.unload()
            else:
                self.challenger_client.stop_server()
            self._save_final_results()

    def _bootstrap(self) -> float:
        """
        Generate initial dataset and evaluate baseline performance.

        Returns:
            Initial solver performance score
        """
        # Generate initial tasks
        logger.info(f"Generating {self.bootstrap_size} bootstrap tasks")
        print(f"\n      === BOOTSTRAP PHASE ===")
        print(f"      Generating {self.bootstrap_size} initial tasks using Challenger (30B model)")
        print(f"      This is a one-time baseline evaluation (no training)")
        print(f"      ")
        tasks = self.challenger.generate_task_batch(
            self.bootstrap_size,
            current_solver_performance=0.5
        )
        print(f"      Successfully generated {len(tasks)} tasks!")

        # Generate pseudo-labels
        logger.info("Generating pseudo-labels with challenger")
        pseudo_labels = self.challenger.generate_pseudo_labels(tasks)

        # Evaluate initial solver performance
        eval_size = int(len(tasks) * self.eval_ratio)
        eval_tasks = tasks[:eval_size]
        eval_labels = pseudo_labels[:eval_size]

        metrics = self.trainer.evaluate(eval_tasks, eval_labels)
        logger.info(f"Bootstrap evaluation score: {metrics['eval_score']:.3f}")

        return metrics['eval_score']

    def _evolution_iteration(self, current_performance: float) -> Dict[str, float]:
        """
        Run one complete evolution iteration.

        Args:
            current_performance: Current solver success rate

        Returns:
            Iteration metrics
        """
        iteration_start = time.time()

        # Step 1: Challenger generates tasks
        logger.info(f"Challenger generating {self.dataset_size} tasks")
        tasks = self.challenger.generate_task_batch(
            self.dataset_size,
            current_solver_performance=current_performance
        )

        # Step 2: Generate pseudo-labels
        logger.info("Generating pseudo-labels")
        pseudo_labels = self.challenger.generate_pseudo_labels(tasks)

        # Step 3: Split into train/eval
        eval_size = int(len(tasks) * self.eval_ratio)
        train_tasks = tasks[eval_size:]
        train_labels = pseudo_labels[eval_size:]
        eval_tasks = tasks[:eval_size]
        eval_labels = pseudo_labels[:eval_size]

        # Step 4: Evaluate challenger quality
        logger.info("Evaluating task quality")
        solver_outputs = []
        for task in train_tasks[:10]:  # Sample for quality check
            if task.metadata['task_type'] == 'code_documentation':
                prompt = f"Generate documentation for:\n{task.input_text}"
            else:
                prompt = f"Task: {task.input_text}"

            output = self.solver_model.generate(
                prompt,
                max_tokens=256,
                temperature=0.7
            )[0]
            solver_outputs.append(output)

        quality_scores = self.challenger.evaluate_solver_outputs(
            train_tasks[:10],
            solver_outputs
        )
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Step 5: Train solver with GRPO
        logger.info("Training solver with GRPO")
        training_metrics = self.trainer.train_iteration(
            train_tasks,
            train_labels,
            [avg_quality] * len(train_tasks)  # Use average quality as base score
        )

        # Step 6: Evaluate solver
        logger.info("Evaluating solver performance")
        eval_metrics = self.trainer.evaluate(eval_tasks, eval_labels)

        # Combine metrics
        metrics = {
            **training_metrics,
            **eval_metrics,
            'task_quality': avg_quality,
            'difficulty': self.challenger.difficulty,
            'iteration_time': time.time() - iteration_start
        }

        # Save all outputs for review
        self._save_solver_outputs(solver_outputs, f'generation_{self.generation}')

        logger.info(f"Iteration complete. Eval score: {eval_metrics['eval_score']:.3f}")
        return metrics

    def _save_generation_data(self, tasks: List, labels: List[str], outputs: Optional[List] = None, prefix: str = 'generation'):
        """Save all data from a generation for review and analysis."""
        data_dir = self.output_dir / 'data' / prefix
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save tasks as pickle (preserves full object)
        with open(data_dir / 'tasks.pkl', 'wb') as f:
            pickle.dump(tasks, f)

        # Save tasks as JSON for readability
        tasks_json = [
            {
                'id': task.id,
                'input_text': task.input_text,
                'difficulty': task.difficulty,
                'metadata': task.metadata
            }
            for task in tasks
        ]
        with open(data_dir / 'tasks.json', 'w') as f:
            json.dump(tasks_json, f, indent=2)

        # Save labels
        with open(data_dir / 'labels.json', 'w') as f:
            json.dump(labels, f, indent=2)

        # Save combined view for easy review
        combined_data = []
        for i, task in enumerate(tasks):
            entry = {
                'task_id': task.id,
                'difficulty': task.difficulty,
                'input_preview': task.input_text[:200] + '...' if len(task.input_text) > 200 else task.input_text,
                'label_preview': labels[i][:200] + '...' if i < len(labels) and len(labels[i]) > 200 else labels[i] if i < len(labels) else None
            }
            combined_data.append(entry)

        with open(data_dir / 'combined_view.json', 'w') as f:
            json.dump(combined_data, f, indent=2)

        logger.info(f"Saved {len(tasks)} tasks and labels to {data_dir}")

    def _save_solver_outputs(self, outputs: List[str], prefix: str):
        """Save solver outputs separately for analysis."""
        output_dir = self.output_dir / 'solver_outputs'
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / f'{prefix}_outputs.json', 'w') as f:
            json.dump(outputs, f, indent=2)

        logger.info(f"Saved {len(outputs)} solver outputs to {output_dir}")

    def _save_best_checkpoint(self):
        """Save the best performing model checkpoint."""
        checkpoint_dir = self.output_dir / 'best_checkpoint'
        checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Saving best checkpoint (score: {self.best_score:.3f})")

        # Save LoRA adapter
        self.solver_model.save_adapter(checkpoint_dir / 'adapter')

        # Save metadata
        metadata = {
            'generation': self.generation,
            'best_score': self.best_score,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_generation_checkpoint(self, metrics: Dict[str, float]):
        """Save checkpoint for current generation."""
        checkpoint_dir = self.output_dir / f'generation_{self.generation:03d}'
        checkpoint_dir.mkdir(exist_ok=True)

        # Save adapter
        self.solver_model.save_adapter(checkpoint_dir / 'adapter')

        # Save metrics
        with open(checkpoint_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved generation checkpoint to {checkpoint_dir}")

    def _save_final_results(self):
        """Save final evolution results and history."""
        results = {
            'total_generations': self.generation,
            'best_score': self.best_score,
            'history': self.history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / 'evolution_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Create summary report
        self._generate_summary_report()

        logger.info(f"Evolution complete. Results saved to {self.output_dir}")

    def _generate_summary_report(self):
        """Generate human-readable summary report."""
        report = []
        report.append("=" * 60)
        report.append("LoRA Evolution Summary Report")
        report.append("=" * 60)
        report.append(f"Output Directory: {self.output_dir}")
        report.append(f"Task Type: {self.config['task']['type']}")
        report.append(f"Total Generations: {self.generation}")
        report.append(f"Best Score: {self.best_score:.4f}")
        report.append("")

        report.append("Evolution Progress:")
        report.append("-" * 40)
        for entry in self.history:
            gen = entry['generation']
            score = entry['metrics']['eval_score']
            difficulty = entry['metrics'].get('difficulty', 0)
            report.append(f"Generation {gen:3d}: Score={score:.3f}, Difficulty={difficulty:.2f}")

        report.append("")
        report.append("Configuration:")
        report.append("-" * 40)
        report.append(f"Challenger Model: {self.config['challenger']['model_path']}")
        report.append(f"Solver Model: {self.config['solver']['model_name']}")
        report.append(f"LoRA Rank: {self.config['solver']['lora_config']['rank']}")
        report.append(f"Learning Rate: {self.config['training']['learning_rate']}")

        report_text = "\n".join(report)

        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)

        print("\n" + report_text)


def load_evolution_checkpoint(checkpoint_dir: str, config: Dict[str, Any]) -> LoRAEvolution:
    """
    Load evolution from checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint
        config: Configuration to use

    Returns:
        LoRAEvolution instance with loaded state
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load metadata
    with open(checkpoint_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Create evolution instance
    evolution = LoRAEvolution(config)

    # Load adapter
    evolution.solver_model.load_adapter(
        checkpoint_dir / 'adapter',
        adapter_name='checkpoint'
    )
    evolution.solver_model.set_adapter('checkpoint')

    # Restore state
    evolution.generation = metadata['generation']
    evolution.best_score = metadata['best_score']

    logger.info(f"Loaded checkpoint from generation {evolution.generation}")
    return evolution
