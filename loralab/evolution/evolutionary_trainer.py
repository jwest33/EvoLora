"""Main evolutionary trainer for self-supervised LoRA optimization

Orchestrates the evolution process to find optimal LoRA configurations.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import json
from tqdm import tqdm

from ..core.model_manager import ModelManager, ModelCheckpoint
from ..core.lora_factory import LoRAFactory, LoRAVariant
from .population import PopulationManager
from .fitness_evaluator import FitnessEvaluator
from ..training.self_supervised import SelfSupervisedTrainer
from ..utils.cli_formatter import CLIFormatter
from ..utils.output_manager import get_output_manager

logger = logging.getLogger(__name__)


class EvolutionaryTrainer:
    """Main trainer for evolutionary LoRA optimization"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration

        Args:
            config: Full configuration dictionary containing:
                - model: Model configuration
                - evolution: Evolution parameters
                - lora_search_space: LoRA hyperparameter search space
                - training: Training parameters
                - dataset: Dataset configuration
                - output: Output configuration (or output_dir for legacy)
        """
        self.config = config

        # Initialize output manager
        output_config = config.get('output', {})
        self.output_manager = get_output_manager(
            base_dir=output_config.get('base_dir', 'lora_runs'),
            run_name=output_config.get('run_name')
        )
        self.output_dir = self.output_manager.get_path('base')
        self.checkpoint_dir = self.output_manager.get_path('checkpoints')

        # Initialize components
        self.model_manager = ModelManager(config['model'])
        self.lora_factory = LoRAFactory(config['lora_search_space'])
        self.population_manager = PopulationManager()
        self.trainer = None  # Will be initialized after model loads
        self.evaluator = None  # Will be initialized after model loads

        # Evolution state
        self.current_generation = 0
        self.best_variant_overall = None
        self.evolution_history = []

    def initialize(self):
        """Initialize the training system"""
        CLIFormatter.print_header("INITIALIZING EVOLUTIONARY LORA TRAINER")

        # Save configuration for this run
        self.output_manager.save_config(self.config)
        CLIFormatter.print_info(f"Run directory: {self.output_manager.get_path('base')}")

        # Load base model
        CLIFormatter.print_info("Loading base model...")
        self.model_manager.load_base_model()

        # Initialize trainer and evaluator with loaded model
        self.trainer = SelfSupervisedTrainer(
            model_manager=self.model_manager,
            training_config=self.config['training']
        )

        self.evaluator = FitnessEvaluator(
            model_manager=self.model_manager
        )

        CLIFormatter.print_success("Initialization complete")

    def train_variant(self,
                     variant: LoRAVariant,
                     train_data: List[Dict],
                     epochs: int = 1) -> float:
        """Train a single LoRA variant

        Args:
            variant: LoRA variant to train
            train_data: Training dataset
            epochs: Number of epochs to train

        Returns:
            Average training loss
        """
        start_time = time.time()

        # Create LoRA model for this variant
        lora_model = self.model_manager.create_lora_variant({
            'rank': variant.rank,
            'alpha': variant.alpha,
            'dropout': variant.dropout,
            'target_modules': variant.target_modules
        })

        variant.model = lora_model

        # Train the variant
        avg_loss = self.trainer.train(
            model=lora_model,
            train_data=train_data,
            learning_rate=variant.learning_rate,
            epochs=epochs,
            variant_id=variant.variant_id
        )

        variant.train_loss = avg_loss
        variant.training_time = time.time() - start_time

        return avg_loss

    def evaluate_variant(self,
                        variant: LoRAVariant,
                        eval_data: List[Dict]) -> Dict[str, float]:
        """Evaluate a variant's performance

        Args:
            variant: LoRA variant to evaluate
            eval_data: Evaluation dataset

        Returns:
            Dictionary with evaluation metrics
        """
        if variant.model is None:
            raise ValueError(f"Variant {variant.variant_id} has no trained model")

        metrics = self.evaluator.evaluate(
            model=variant.model,
            eval_data=eval_data,
            variant_id=variant.variant_id
        )

        # Update variant metrics
        variant.eval_accuracy = metrics['accuracy']
        variant.eval_perplexity = metrics['perplexity']

        return metrics

    def run_generation(self,
                      population: List[LoRAVariant],
                      train_data: List[Dict],
                      eval_data: List[Dict]) -> List[LoRAVariant]:
        """Run one generation of evolution

        Args:
            population: Current population of variants
            train_data: Training dataset
            eval_data: Evaluation dataset

        Returns:
            Selected survivors for next generation
        """
        CLIFormatter.print_generation_header(
            self.current_generation,
            self.config['evolution']['generations']
        )

        generation_start = time.time()

        # Train and evaluate each variant
        for i, variant in enumerate(population):
            CLIFormatter.print_subheader(
                f"Variant {i+1}/{len(population)}: {variant.variant_id}"
            )

            # Show configuration
            CLIFormatter.print_variant_status(variant.variant_id, "TRAINING", {
                'Rank': variant.rank,
                'Learning Rate': f"{float(variant.learning_rate):.0e}",
                'Dropout': variant.dropout
            })

            # Train
            train_loss = self.train_variant(
                variant,
                train_data,
                epochs=self.config['training'].get('epochs_per_variant', 1)
            )

            # Evaluate
            CLIFormatter.print_variant_status(variant.variant_id, "EVALUATING")
            metrics = self.evaluate_variant(variant, eval_data)

            # Show results
            CLIFormatter.print_variant_status(variant.variant_id, "COMPLETE", {
                'Training Loss': train_loss,
                'Accuracy': f"{metrics['accuracy']:.2%}",
                'Perplexity': metrics['perplexity']
            })

            # Save checkpoint for this variant
            self._save_variant_checkpoint(variant)

            # Clean up model to save memory
            if i < len(population) - 1:  # Keep last model for potential reuse
                del variant.model
                variant.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Select survivors
        keep_top = self.config['evolution'].get('keep_top', 2)
        survivors = self.population_manager.select_survivors(population, keep_top)

        # Update best overall
        if survivors and (self.best_variant_overall is None or
                         survivors[0].fitness_score() > self.best_variant_overall.fitness_score()):
            self.best_variant_overall = survivors[0]
            self._save_best_variant(survivors[0])

        # Log generation summary
        generation_time = time.time() - generation_start
        self._log_generation_summary(population, survivors, generation_time)

        # Save generation history
        self._save_generation_history(population)

        return survivors

    def evolve(self,
              train_data: List[Dict],
              eval_data: List[Dict],
              resume_from: Optional[str] = None) -> LoRAVariant:
        """Run the full evolution process

        Args:
            train_data: Training dataset
            eval_data: Evaluation dataset
            resume_from: Path to checkpoint to resume from

        Returns:
            Best variant found
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING EVOLUTIONARY OPTIMIZATION")
        logger.info("="*80)

        # Resume or start fresh
        if resume_from and Path(resume_from).exists():
            population, start_gen = self._load_checkpoint(resume_from)
            self.current_generation = start_gen
            logger.info(f"Resumed from generation {start_gen}")
        else:
            # Create initial population
            population_size = self.config['evolution']['population_size']
            population = self.lora_factory.create_population(population_size)
            self.current_generation = 0

        # Evolution loop
        num_generations = self.config['evolution']['generations']

        for gen in range(self.current_generation, num_generations):
            self.current_generation = gen

            # Run generation
            survivors = self.run_generation(population, train_data, eval_data)

            # Save checkpoint
            self._save_checkpoint(survivors)

            # Create next generation (unless last generation)
            if gen < num_generations - 1:
                population = self.lora_factory.evolve_population(
                    survivors=survivors,
                    population_size=self.config['evolution']['population_size'],
                    mutation_rate=self.config['evolution'].get('mutation_rate', 0.3)
                )
            else:
                population = survivors

        # Final summary
        self._print_final_summary()

        # Generate comparison report for best variant if requested
        if self.config.get('generate_comparison_report', True) and self.best_variant_overall:
            self._generate_comparison_report(eval_data[:100])  # Use subset for report

        return self.best_variant_overall

    def _save_variant_checkpoint(self, variant: LoRAVariant):
        """Save a variant's model and configuration"""
        # Use new structure
        variant_dir = self.output_manager.get_generation_checkpoint_dir(self.current_generation) / variant.variant_id
        model_dir = self.output_manager.get_variant_model_dir(variant.variant_id)

        variant_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration to checkpoint
        self.lora_factory.save_variant(variant, str(variant_dir / "config.json"))

        # Save model to models directory
        if variant.model is not None:
            variant.model.save_pretrained(str(model_dir / "adapter"))

    def _save_best_variant(self, variant: LoRAVariant):
        """Save the best variant found so far"""
        best_dir = self.output_manager.get_path('best_model')
        best_dir.mkdir(exist_ok=True)

        # Save configuration
        self.lora_factory.save_variant(variant, str(best_dir / "config.json"))

        # Save model if it exists
        if variant.model is not None:
            variant.model.save_pretrained(str(best_dir / "adapter"))
            self.model_manager.get_tokenizer().save_pretrained(str(best_dir / "adapter"))

        logger.info(f"✓ New best variant saved: {variant.variant_id} "
                   f"(accuracy: {variant.eval_accuracy:.2%})")

    def _save_checkpoint(self, survivors: List[LoRAVariant]):
        """Save evolution checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"evolution_gen{self.current_generation}.json"

        checkpoint_data = {
            'generation': self.current_generation,
            'survivors': [v.to_dict() for v in survivors],
            'best_overall': self.best_variant_overall.to_dict() if self.best_variant_overall else None,
            'history': self.evolution_history
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load evolution checkpoint"""
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        # Reconstruct survivors
        survivors = [LoRAVariant.from_dict(v) for v in data['survivors']]

        # Reconstruct best overall
        if data['best_overall']:
            self.best_variant_overall = LoRAVariant.from_dict(data['best_overall'])

        # Restore history
        self.evolution_history = data.get('history', [])

        return survivors, data['generation']

    def _save_generation_history(self, population: List[LoRAVariant]):
        """Save generation statistics to history"""
        stats = {
            'generation': self.current_generation,
            'population_size': len(population),
            'best_accuracy': max(v.eval_accuracy for v in population),
            'avg_accuracy': sum(v.eval_accuracy for v in population) / len(population),
            'best_perplexity': min(v.eval_perplexity for v in population),
            'variants': [v.to_dict() for v in population]
        }

        self.evolution_history.append(stats)

        # Save to file
        history_path = self.output_manager.save_history(self.evolution_history)

    def _log_generation_summary(self,
                               population: List[LoRAVariant],
                               survivors: List[LoRAVariant],
                               generation_time: float):
        """Log generation summary"""
        CLIFormatter.print_subheader(f"Generation {self.current_generation} Summary")

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda v: v.fitness_score(), reverse=True)

        # Prepare table data
        headers = ["Rank", "Variant ID", "Accuracy", "Perplexity", "Fitness", "Status"]
        rows = []
        colors = []

        for i, variant in enumerate(sorted_pop):
            if variant in survivors:
                status = "SURVIVED"
                from ..utils.cli_formatter import Fore
                colors.append(Fore.GREEN)
            else:
                status = "ELIMINATED"
                from ..utils.cli_formatter import Fore
                colors.append(Fore.RED)

            rows.append([
                f"#{i+1}",
                variant.variant_id,
                f"{variant.eval_accuracy:.2%}",
                f"{variant.eval_perplexity:.2f}",
                f"{variant.fitness_score():.4f}",
                status
            ])

        CLIFormatter.print_table(headers, rows, colors)

        # Time summary
        CLIFormatter.print_info(f"Generation completed in {CLIFormatter.format_time(generation_time)}")

    def _print_final_summary(self):
        """Print final evolution summary"""
        CLIFormatter.print_header("EVOLUTION COMPLETE", char="=")

        if self.best_variant_overall:
            # Configuration summary
            config_summary = {
                'Variant ID': self.best_variant_overall.variant_id,
                'Generation': self.best_variant_overall.generation,
                'LoRA Rank': self.best_variant_overall.rank,
                'Alpha': self.best_variant_overall.alpha,
                'Dropout': self.best_variant_overall.dropout,
                'Learning Rate': f"{float(self.best_variant_overall.learning_rate):.0e}"
            }

            CLIFormatter.print_summary_box("BEST VARIANT CONFIGURATION", config_summary)

            # Performance summary
            perf_summary = {
                'Accuracy': self.best_variant_overall.eval_accuracy,
                'Perplexity': self.best_variant_overall.eval_perplexity,
                'Fitness Score': self.best_variant_overall.fitness_score()
            }

            from ..utils.cli_formatter import Fore
            CLIFormatter.print_summary_box("PERFORMANCE METRICS", perf_summary, color=Fore.CYAN)

            CLIFormatter.print_success(f"Best variant saved to: {self.output_dir / 'best_variant'}")

    def _generate_comparison_report(self, eval_data: List[Dict]):
        """Generate comparison report for best variant

        Args:
            eval_data: Evaluation dataset
        """
        try:
            from ..evaluation.comparative_evaluator import ComparativeEvaluator

            logger.info("\nGenerating comparison report for best variant...")

            # Load best variant model
            best_dir = self.output_manager.get_path('best_model') / "adapter"
            if not best_dir.exists():
                logger.warning("Best variant adapter not found, skipping report")
                return

            # Create base model for comparison
            base_lora_config = {
                'rank': self.best_variant_overall.rank,
                'alpha': self.best_variant_overall.alpha,
                'dropout': self.best_variant_overall.dropout,
                'target_modules': self.best_variant_overall.target_modules
            }
            base_model = self.model_manager.create_lora_variant(base_lora_config)

            # Load trained adapter
            from peft import PeftModel
            lora_model = PeftModel.from_pretrained(
                self.model_manager.base_model,
                str(best_dir)
            )

            # Generate comparison
            comparator = ComparativeEvaluator(
                self.model_manager,
                output_dir=str(self.output_manager.get_path('reports'))
            )

            comparison = comparator.compare_models(
                base_model=base_model,
                lora_model=lora_model,
                eval_data=eval_data
            )

            report_path = comparator.generate_report(
                comparison,
                self.best_variant_overall.variant_id
            )

            logger.info(f"✓ Comparison report saved to: {report_path}")

            # Print summary
            summary = comparison['summary']
            logger.info("\nComparison Summary:")
            logger.info(f"  Base accuracy: {summary['base_accuracy']:.2%}")
            logger.info(f"  LoRA accuracy: {summary['lora_accuracy']:.2%}")
            logger.info(f"  Improvement: {summary['improvement']:.2%}")
            logger.info(f"  Improvements: {summary['improvements_count']} examples")

        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
