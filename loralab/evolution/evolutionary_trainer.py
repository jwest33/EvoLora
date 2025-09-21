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
import os

# Enable logits for Unsloth models during evaluation
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from ..core.model_manager import ModelManager, ModelCheckpoint
from ..core.lora_factory import LoRAFactory, LoRAVariant
from .population import PopulationManager
from .fitness_evaluator import FitnessEvaluator
from ..training.self_supervised import SelfSupervisedTrainer
from ..utils.cli_formatter import CLIFormatter
from ..utils.output_manager import get_output_manager
from ..utils.memory_optimizer import optimize_memory, cleanup_after_variant

# Import new components
try:
    from ..core.unsloth_manager import UnslothModelManager, create_model_manager
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    create_model_manager = None

try:
    from ..training.unsloth_sft_trainer import UnslothSFTTrainer, create_trainer
    from ..training.grpo_trainer import GRPOTrainer
    ADVANCED_TRAINERS = True
except ImportError:
    ADVANCED_TRAINERS = False
    create_trainer = None

try:
    from ..utils.model_exporter import ModelExporter
    EXPORTER_AVAILABLE = True
except ImportError:
    EXPORTER_AVAILABLE = False

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
        # Choose model manager based on backend
        if UNSLOTH_AVAILABLE and create_model_manager:
            self.model_manager = create_model_manager(config['model'])
            logger.info(f"Using {config['model'].get('backend', 'transformers')} backend")
        else:
            self.model_manager = ModelManager(config['model'])
            logger.info("Using standard transformers backend")

        self.lora_factory = LoRAFactory(config['lora_search_space'])
        self.population_manager = PopulationManager()
        self.trainer = None  # Will be initialized after model loads
        self.evaluator = None  # Will be initialized after model loads
        self.exporter = None  # Model export utilities

        # Evolution state
        self.current_generation = 0
        self.best_variant_overall = None
        self.evolution_history = []

    def initialize(self):
        """Initialize the training system"""
        CLIFormatter.print_header("INITIALIZING EVOLORA")

        # Save configuration for this run
        self.output_manager.save_config(self.config)
        CLIFormatter.print_info(f"Run directory: {self.output_manager.get_path('base')}")

        # Optimize memory before loading model
        optimize_memory()

        # Load base model
        CLIFormatter.print_info("Loading base model...")
        self.model_manager.load_base_model()

        # Initialize trainer based on method
        training_method = self.config['training'].get('method', 'sft')

        if training_method == 'grpo' and ADVANCED_TRAINERS:
            # Use GRPO trainer for reasoning tasks
            CLIFormatter.print_info("Using GRPO trainer for reasoning tasks")
            # Merge GRPO config into training config for trainer
            training_config_with_grpo = self.config['training'].copy()
            training_config_with_grpo['grpo'] = self.config.get('grpo', {})
            self.trainer = GRPOTrainer(
                model_manager=self.model_manager,
                training_config=training_config_with_grpo,
                full_config=self.config
            )
        elif ADVANCED_TRAINERS and create_trainer:
            # Use factory to create appropriate trainer
            CLIFormatter.print_info(f"Using {training_method.upper()} training method")
            self.trainer = create_trainer(
                model_manager=self.model_manager,
                training_config=self.config['training'],
                full_config=self.config
            )
        else:
            # Fall back to original self-supervised trainer
            CLIFormatter.print_info("Using standard self-supervised trainer")
            self.trainer = SelfSupervisedTrainer(
                model_manager=self.model_manager,
                training_config=self.config['training']
            )

        self.evaluator = FitnessEvaluator(
            model_manager=self.model_manager
        )

        # Initialize exporter if available
        if EXPORTER_AVAILABLE:
            self.exporter = ModelExporter(self.model_manager)

        CLIFormatter.print_success("Initialization complete")

    def train_variant(self,
                     variant: LoRAVariant,
                     train_data: List[Dict],
                     epochs: int = 1,
                     variant_num: int = None) -> float:
        """Train a single LoRA variant

        Args:
            variant: LoRA variant to train
            train_data: Training dataset
            epochs: Number of epochs to train
            variant_num: Variant number within generation (for job tracking)

        Returns:
            Average training loss or fitness score
        """
        start_time = time.time()


        # Create LoRA model for this variant with enhanced config
        lora_config = {
            'rank': variant.rank,
            'alpha': variant.alpha,
            'dropout': variant.dropout,
            'target_modules': variant.target_modules
        }

        # Add Unsloth-specific options if available
        if isinstance(self.model_manager, UnslothModelManager if UNSLOTH_AVAILABLE else type(None)):
            lora_config['alpha_multiplier'] = variant.alpha // variant.rank if variant.rank > 0 else 2
            lora_config['use_rslora'] = variant.use_rslora
            lora_config['use_gradient_checkpointing'] = self.config['lora_search_space'].get('use_gradient_checkpointing', True)

        lora_model = self.model_manager.create_lora_variant(lora_config)
        variant.model = lora_model

        # Temporarily override training config with variant-specific values
        original_config = self.config['training'].copy()
        self.config['training']['weight_decay'] = variant.weight_decay
        self.config['training']['warmup_ratio'] = variant.warmup_ratio
        self.config['training']['max_grad_norm'] = variant.max_grad_norm

        # Update trainer config if it has a reference to training_config
        if hasattr(self.trainer, 'training_config'):
            self.trainer.training_config = self.config['training']

        # Choose training approach based on trainer type
        training_method = self.config['training'].get('method', 'sft')

        if training_method == 'grpo' and hasattr(self.trainer, 'pre_train_formatting'):
            # GRPO training with pre-training if needed
            grpo_config = self.config.get('grpo', {})

            # Pre-train on format if requested
            if grpo_config.get('pre_train_format', False):
                # Use configured number of format examples (default to 5 for efficiency)
                num_format_examples = grpo_config.get('format_examples', 5)

                logger.info(f"Train data has {len(train_data)} total examples")
                logger.info(f"Requesting {num_format_examples} format examples")

                format_examples = train_data[:num_format_examples]

                logger.info(f"Actually got {len(format_examples)} format examples")

                # Get pre-training epochs from config or use default
                pre_train_epochs = grpo_config.get('pre_train_epochs', 2)

                logger.info(f"Pre-training on format with {len(format_examples)} examples for {pre_train_epochs} epochs")

                self.trainer.pre_train_formatting(
                    model=lora_model,
                    format_examples=format_examples,
                    learning_rate=variant.learning_rate * 2,  # Higher LR for pre-training
                    epochs=pre_train_epochs
                )
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
            # Main GRPO training
            metrics = self.trainer.train(
                model=lora_model,
                train_data=train_data,
                learning_rate=variant.learning_rate,
                max_steps=grpo_config.get('max_steps', 100),
                variant_id=variant.variant_id
            )
            avg_loss = metrics.get('final_loss', float('inf'))
            variant.rewards = metrics.get('rewards', 0.0)

        elif hasattr(self.trainer, 'train_on_responses'):
            # TRL SFTTrainer with advanced options
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
            metrics = self.trainer.train(
                model=lora_model,
                train_data=train_data,
                learning_rate=variant.learning_rate,
                epochs=epochs,
                variant_id=variant.variant_id,
                train_on_responses=self.config['training'].get('train_on_completions_only', False)
            )
            avg_loss = metrics.get('final_loss', float('inf'))
        
        else:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
            # Standard training
            avg_loss = self.trainer.train(
                model=lora_model,
                train_data=train_data,
                learning_rate=variant.learning_rate,
                epochs=epochs,
                variant_id=variant.variant_id
            )

        variant.train_loss = avg_loss
        variant.training_time = time.time() - start_time

        # Restore original training config
        self.config['training'] = original_config
        if hasattr(self.trainer, 'training_config'):
            self.trainer.training_config = original_config

        return avg_loss

    def evaluate_variant(self,
                        variant: LoRAVariant,
                        eval_data: List[Dict],
                        fast_mode: bool = True) -> Dict[str, float]:
        """Evaluate a variant's performance

        Args:
            variant: LoRA variant to evaluate
            eval_data: Evaluation dataset
            fast_mode: If True, use fast evaluation (perplexity only)

        Returns:
            Dictionary with evaluation metrics
        """
        if variant.model is None:
            raise ValueError(f"Variant {variant.variant_id} has no trained model")
        
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        metrics = self.evaluator.evaluate(
            model=variant.model,
            eval_data=eval_data,
            variant_id=variant.variant_id,
            fast_mode=fast_mode
        )

        # Update variant metrics
        variant.eval_accuracy = metrics.get('accuracy', 0.0)
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
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        CLIFormatter.print_generation_header(
            self.current_generation,
            self.config['evolution']['generations']
        )

        generation_start = time.time()

        # Sequential processing
        processed_population = []

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
                epochs=self.config['training'].get('epochs_per_variant', 1),
                variant_num=i + 1
            )

            # Evaluate
            CLIFormatter.print_variant_status(variant.variant_id, "EVALUATING")
            # Disable fast mode to actually calculate accuracy
            fast_mode = False  # Changed: We need accuracy for GSM8K
            metrics = self.evaluate_variant(variant, eval_data, fast_mode=fast_mode)

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
                # Aggressive memory cleanup after each variant
                cleanup_after_variant()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            processed_population.append(variant)

        # Select survivors
        keep_top = self.config['evolution'].get('keep_top', 2)
        survivors = self.population_manager.select_survivors(processed_population, keep_top)

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
            self._generate_comparison_report()  # Uses fixed evaluation set

        return self.best_variant_overall

    def _save_variant_checkpoint(self, variant: LoRAVariant):
        """Save a variant's model and configuration"""
        # Use new structure
        variant_dir = self.output_manager.get_generation_checkpoint_dir(self.current_generation) / variant.variant_id
        model_dir = self.output_manager.get_variant_model_dir(variant.variant_id)

        # Save configuration to checkpoint (directory already created by get_generation_checkpoint_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)  # Still need this for the variant subdirectory
        self.lora_factory.save_variant(variant, str(variant_dir / "config.json"))

        # Save model to models directory
        if variant.model is not None:
            variant.model.save_pretrained(str(model_dir / "adapter"))

    def _save_best_variant(self, variant: LoRAVariant):
        """Save the best variant found so far"""
        best_dir = self.output_manager.get_path('best_model')  # Directory created by get_path

        # Save configuration
        self.lora_factory.save_variant(variant, str(best_dir / "config.json"))

        # Save model - reload if necessary
        if variant.model is None:
            # Model was cleaned up, need to reload it to save
            logger.debug(f"Reloading model for best variant {variant.variant_id} to save adapter")
            # Try to load from checkpoint if it exists
            checkpoint_path = self.checkpoint_dir / f"generations/gen{variant.generation}/{variant.variant_id}/adapter"
            if checkpoint_path.exists():
                from peft import PeftModel
                base_model = self.model_manager.get_model()
                variant.model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
                logger.debug(f"Loaded adapter from checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"Could not find checkpoint for {variant.variant_id}, skipping adapter save")

        if variant.model is not None:
            variant.model.save_pretrained(str(best_dir / "adapter"))
            self.model_manager.get_tokenizer().save_pretrained(str(best_dir / "adapter"))

            # Export in additional formats if exporter is available
            if self.exporter and EXPORTER_AVAILABLE:
                export_config = self.config.get('export', {})
                if export_config.get('auto_export', False):
                    formats = export_config.get('formats', ['lora'])
                    variant_info = {
                        'variant_id': variant.variant_id,
                        'rank': variant.rank,
                        'alpha': variant.alpha,
                        'generation': self.current_generation,
                        'accuracy': variant.eval_accuracy,
                        'fitness': variant.fitness_score()
                    }

                    export_paths = self.exporter.export_best_variant(
                        model=variant.model,
                        variant_info=variant_info,
                        output_dir=str(best_dir.parent / 'exports'),
                        formats=formats
                    )

                    logger.info(f"Exported best model in formats: {list(export_paths.keys())}")

        logger.info(f"[+] New best variant saved: {variant.variant_id} "
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

    def _generate_comparison_report(self, eval_data: Optional[List[Dict]] = None):
        """Generate comparison report for best variant using fixed evaluation set

        Args:
            eval_data: Optional evaluation dataset (if None, uses fixed set)
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

            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
            
            lora_model = PeftModel.from_pretrained(
                self.model_manager.get_model(),
                str(best_dir)
            )

            # Generate comparison
            reports_dir = self.output_manager.get_path('base') / 'reports'
            reports_dir.mkdir(exist_ok=True)
            comparator = ComparativeEvaluator(
                self.model_manager,
                output_dir=str(reports_dir)
            )

            # Use fixed evaluation set by default
            comparison = comparator.compare_models(
                base_model=base_model,
                lora_model=lora_model,
                eval_data=eval_data,
                use_fixed_set=(eval_data is None)  # Use fixed set if no data provided
            )

            report_path = comparator.generate_report(
                comparison,
                self.best_variant_overall.variant_id
            )

            logger.info(f"[SUCCESS] Comparison report saved to: {report_path}")

            # Print summary
            summary = comparison['summary']
            logger.info("\nComparison Summary:")
            logger.info(f"  Base accuracy: {summary['base_accuracy']:.2%}")
            logger.info(f"  LoRA accuracy: {summary['lora_accuracy']:.2%}")
            logger.info(f"  Improvement: {summary['improvement']:.2%}")
            logger.info(f"  Improvements: {summary['improvements_count']} examples")

        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
