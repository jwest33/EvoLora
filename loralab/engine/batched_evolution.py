"""
Memory-efficient batched co-evolution engine.
Runs Challenger and Solver in separate phases to avoid loading both models simultaneously.
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
import gc

from loralab.core.llama_client import LlamaCppClient
from loralab.core.llama_direct import LlamaDirectClient
from loralab.core.solver_client import SolverModel
from loralab.generation.task_challenger import TaskChallenger
from loralab.adaptation.lora_solver import LoRASolverTrainer

logger = logging.getLogger(__name__)


class BatchedEvolution:
    """Memory-efficient evolution that runs models in batches."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batched evolution engine.

        Args:
            config: Full configuration including challenger, solver, and training params
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create batch directories
        self.batch_dir = self.output_dir / 'batches'
        self.batch_dir.mkdir(exist_ok=True)

        # Evolution parameters
        self.num_generations = config['evolution'].get('generations', 50)
        self.dataset_size = config['evolution'].get('dataset_size_per_gen', 20)
        self.bootstrap_size = config['evolution'].get('bootstrap_size', 10)
        self.eval_ratio = config['evolution'].get('eval_ratio', 0.2)

        # Tracking
        self.generation = 0
        self.best_score = -float('inf')
        self.history = []

    def run(self):
        """Run the complete evolution process with batched execution."""
        print(f"\n{'='*60}")
        print(f"MEMORY-EFFICIENT BATCHED EVOLUTION")
        print(f"{'='*60}")
        print(f"Generations: {self.num_generations}")
        print(f"Tasks per generation: {self.dataset_size}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        try:
            # PHASE 1: Bootstrap with Challenger
            print("[PHASE 1] Bootstrap - Challenger Only")
            print("-" * 40)
            bootstrap_data = self._run_challenger_bootstrap()

            # Get initial baseline (no training needed)
            solver_performance = self._evaluate_baseline(bootstrap_data)
            print(f"Initial solver baseline: {solver_performance:.1%}\n")

            # Evolution loop
            for gen in range(self.num_generations):
                self.generation = gen + 1
                print(f"\n{'='*60}")
                print(f"GENERATION {self.generation}/{self.num_generations}")
                print(f"Current Performance: {solver_performance:.1%}")
                print(f"{'='*60}")

                # PHASE 2: Generate tasks and labels with Challenger
                print(f"\n[PHASE 2.{gen+1}] Challenger Batch")
                print("-" * 40)
                generation_data = self._run_challenger_batch(solver_performance)

                # PHASE 3: Train and evaluate with Solver
                print(f"\n[PHASE 3.{gen+1}] Solver Batch")
                print("-" * 40)
                metrics = self._run_solver_batch(generation_data)

                # Update performance
                solver_performance = metrics['eval_score']

                # Save checkpoint if improved
                if solver_performance > self.best_score:
                    print(f"\nNew best score: {solver_performance:.3f}")
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

                # Early stopping
                if solver_performance > 0.95:
                    print(f"\nEarly stopping: Achieved {solver_performance:.1%} performance!")
                    break

        finally:
            self._save_final_results()
            print(f"\n{'='*60}")
            print(f"Evolution complete! Results saved to:")
            print(f"  {self.output_dir}")
            print(f"{'='*60}\n")

    def _run_challenger_bootstrap(self) -> Dict[str, Any]:
        """
        Run Challenger to generate bootstrap tasks and labels.
        Returns data dict that is saved to disk.
        """
        print("Starting Challenger for bootstrap...")

        # Initialize Challenger (use direct mode if configured)
        if self.config['challenger'].get('use_direct', False):
            print("Using direct llama-cpp-python (no server)")
            challenger_client = LlamaDirectClient(self.config['challenger'])
        else:
            challenger_client = LlamaCppClient(self.config['challenger'])
            # Start server only if not using direct mode
            challenger_client.start_server()

        challenger = TaskChallenger(challenger_client, self.config['task'])

        try:
            # Generate bootstrap tasks
            print(f"Generating {self.bootstrap_size} bootstrap tasks...")
            tasks = challenger.generate_task_batch(
                self.bootstrap_size,
                current_solver_performance=0.5
            )

            # Generate pseudo-labels
            print(f"Generating pseudo-labels...")
            pseudo_labels = challenger.generate_pseudo_labels(tasks)

            # Save to disk
            bootstrap_file = self.batch_dir / 'bootstrap.pkl'
            data = {
                'tasks': tasks,
                'labels': pseudo_labels,
                'generation': 0
            }
            with open(bootstrap_file, 'wb') as f:
                pickle.dump(data, f)

            print(f"Saved bootstrap data to {bootstrap_file}")
            return data

        finally:
            # Clean up Challenger
            if self.config['challenger'].get('use_direct', False):
                challenger_client.unload()
            else:
                challenger_client.stop_server()
            del challenger
            del challenger_client
            gc.collect()
            torch.cuda.empty_cache()
            print("Challenger unloaded from memory")

    def _run_challenger_batch(self, solver_performance: float) -> Dict[str, Any]:
        """
        Run Challenger to generate tasks and labels for one generation.
        """
        print(f"Starting Challenger for generation {self.generation}...")

        # Initialize Challenger (use direct mode if configured)
        if self.config['challenger'].get('use_direct', False):
            print("Loading Challenger model directly (no server)...")
            challenger_client = LlamaDirectClient(self.config['challenger'])
        else:
            challenger_client = LlamaCppClient(self.config['challenger'])
            # Start server only if not using direct mode
            challenger_client.start_server()

        challenger = TaskChallenger(challenger_client, self.config['task'])

        try:
            # Generate tasks
            print(f"Generating {self.dataset_size} tasks (difficulty adjusted for {solver_performance:.1%})...")
            tasks = challenger.generate_task_batch(
                self.dataset_size,
                current_solver_performance=solver_performance
            )

            # Generate pseudo-labels
            print(f"Generating pseudo-labels...")
            pseudo_labels = challenger.generate_pseudo_labels(tasks)

            # Evaluate task quality (optional)
            print("Evaluating task quality...")
            quality_scores = []
            for task in tasks[:5]:  # Sample a few
                score = challenger._evaluate_documentation(
                    task.input_text,
                    pseudo_labels[tasks.index(task)]
                )
                quality_scores.append(score)
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

            # Save to disk
            batch_file = self.batch_dir / f'generation_{self.generation:03d}.pkl'
            data = {
                'tasks': tasks,
                'labels': pseudo_labels,
                'quality': avg_quality,
                'generation': self.generation
            }
            with open(batch_file, 'wb') as f:
                pickle.dump(data, f)

            print(f"Saved generation data to {batch_file}")
            print(f"Average task quality: {avg_quality:.2f}")
            return data

        finally:
            # Clean up Challenger completely
            if self.config['challenger'].get('use_direct', False):
                challenger_client.unload()
            else:
                challenger_client.stop_server()
            del challenger
            del challenger_client
            gc.collect()
            torch.cuda.empty_cache()
            print("Challenger unloaded from memory")

    def _run_solver_batch(self, generation_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Run Solver training and evaluation on the generated data.
        """
        print(f"Starting Solver for generation {self.generation}...")

        # Initialize Solver
        solver_model = SolverModel(self.config['solver'])
        trainer = LoRASolverTrainer(solver_model, self.config['training'])

        # Load checkpoint if exists
        checkpoint_dir = self.output_dir / f'generation_{self.generation-1:03d}' / 'adapter'
        if checkpoint_dir.exists() and self.generation > 1:
            print(f"Loading checkpoint from generation {self.generation-1}...")
            solver_model.load_adapter(str(checkpoint_dir))

        try:
            # Extract data
            tasks = generation_data['tasks']
            labels = generation_data['labels']
            quality = generation_data.get('quality', 0.5)

            # Split into train/eval
            eval_size = int(len(tasks) * self.eval_ratio)
            train_tasks = tasks[eval_size:]
            train_labels = labels[eval_size:]
            eval_tasks = tasks[:eval_size]
            eval_labels = labels[:eval_size]

            print(f"Training on {len(train_tasks)} tasks, evaluating on {len(eval_tasks)} tasks")

            # Train
            print("Training solver with GRPO...")
            start_time = time.time()
            training_metrics = trainer.train_iteration(
                train_tasks,
                train_labels,
                [quality] * len(train_tasks)
            )
            train_time = time.time() - start_time

            # Evaluate
            print("Evaluating solver...")
            eval_metrics = trainer.evaluate(eval_tasks, eval_labels)

            # Combine metrics
            metrics = {
                **training_metrics,
                **eval_metrics,
                'task_quality': quality,
                'train_time': train_time,
                'num_train_tasks': len(train_tasks),
                'num_eval_tasks': len(eval_tasks)
            }

            print(f"Training complete in {train_time/60:.1f} minutes")
            print(f"Evaluation score: {eval_metrics['eval_score']:.3f}")

            # Save adapter for this generation
            gen_dir = self.output_dir / f'generation_{self.generation:03d}'
            gen_dir.mkdir(exist_ok=True)
            solver_model.save_adapter(gen_dir / 'adapter')

            return metrics

        finally:
            # Clean up Solver completely
            del trainer
            del solver_model
            gc.collect()
            torch.cuda.empty_cache()
            print("Solver unloaded from memory")

    def _evaluate_baseline(self, bootstrap_data: Dict[str, Any]) -> float:
        """
        Evaluate baseline solver performance without training.
        """
        print("Evaluating baseline solver performance...")

        # Initialize Solver
        solver_model = SolverModel(self.config['solver'])
        trainer = LoRASolverTrainer(solver_model, self.config['training'])

        try:
            tasks = bootstrap_data['tasks']
            labels = bootstrap_data['labels']

            # Evaluate on all bootstrap tasks
            metrics = trainer.evaluate(tasks, labels)
            return metrics['eval_score']

        finally:
            # Clean up
            del trainer
            del solver_model
            gc.collect()
            torch.cuda.empty_cache()

    def _save_best_checkpoint(self):
        """Copy best checkpoint (already saved during solver batch)."""
        best_dir = self.output_dir / 'best_checkpoint'
        best_dir.mkdir(exist_ok=True)

        # Copy from current generation
        current_adapter = self.output_dir / f'generation_{self.generation:03d}' / 'adapter'
        if current_adapter.exists():
            import shutil
            shutil.copytree(current_adapter, best_dir / 'adapter', dirs_exist_ok=True)

        # Save metadata
        metadata = {
            'generation': self.generation,
            'score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }
        with open(best_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_generation_checkpoint(self, metrics: Dict[str, float]):
        """Save metrics for current generation."""
        gen_dir = self.output_dir / f'generation_{self.generation:03d}'
        gen_dir.mkdir(exist_ok=True)

        with open(gen_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    def _save_final_results(self):
        """Save final evolution results."""
        results = {
            'total_generations': self.generation,
            'best_score': self.best_score,
            'history': self.history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / 'evolution_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evolution complete. Results saved to {self.output_dir}")
