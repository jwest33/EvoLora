"""Command-line interface for self-supervised LoRA evolution

Main entry point for the evolutionary optimization system.
"""

import argparse
import logging
from pathlib import Path
import sys

from .config.config_loader import ConfigLoader
from .evolution.evolutionary_trainer import EvolutionaryTrainer
from .datasets.dataset_loader import DatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evolve_command(args):
    """Run evolutionary optimization"""
    # Load configuration
    if args.config:
        config = ConfigLoader.load_config(args.config)
    else:
        config = ConfigLoader.load_config('loralab/config/config.yaml')

    # Check mode
    if config.get('mode') != 'self_supervised':
        logger.warning(f"Config mode is '{config.get('mode')}', switching to self_supervised")
        config['mode'] = 'self_supervised'

    # The config is already validated and has defaults from ConfigLoader
    # No need for separate self_supervised section
    ss_config = config

    # Apply quick test overrides if requested
    if args.quick_test:
        logger.info("Running in quick test mode")
        ss_config['evolution']['population_size'] = 2
        ss_config['evolution']['generations'] = 2
        ss_config['dataset']['train_size'] = 100
        ss_config['dataset']['eval_size'] = 20

    # Apply CLI overrides
    if args.generations:
        ss_config['evolution']['generations'] = args.generations
    if args.population:
        ss_config['evolution']['population_size'] = args.population
    if args.train_size:
        ss_config['dataset']['train_size'] = args.train_size
    if args.eval_size:
        ss_config['dataset']['eval_size'] = args.eval_size

    # Initialize trainer
    trainer = EvolutionaryTrainer(ss_config)
    trainer.initialize()

    # Load dataset
    loader = DatasetLoader()
    dataset_name = ss_config['dataset']['sources'][0]
    data = loader.load_dataset(
        dataset_name=dataset_name,
        train_size=ss_config['dataset']['train_size'],
        eval_size=ss_config['dataset']['eval_size']
    )

    # Run evolution
    best_variant = trainer.evolve(
        train_data=data['train'],
        eval_data=data['eval'],
        resume_from=args.resume
    )

    # Report results
    logger.info("\n" + "="*80)
    logger.info("EVOLUTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Best variant: {best_variant.variant_id}")
    logger.info(f"Accuracy: {best_variant.eval_accuracy:.2%}")
    logger.info(f"Configuration: rank={best_variant.rank}, lr={best_variant.learning_rate:.0e}")
    logger.info(f"Results saved to: {ss_config['output_dir']}")


def evaluate_command(args):
    """Evaluate a specific adapter"""
    from .evolution.fitness_evaluator import FitnessEvaluator
    from .evaluation.comparative_evaluator import ComparativeEvaluator
    from .core.model_manager import ModelManager
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # Load configuration
    config = ConfigLoader.load_config(args.config if args.config else 'loralab/config/config.yaml')

    model_config = config.get('model', config.get('self_supervised', {}).get('model', {}))
    if not model_config:
        logger.error("No model configuration found")
        sys.exit(1)

    # Load base model
    logger.info("Loading base model...")
    manager = ModelManager(model_config)
    manager.load_base_model()

    # Load adapter
    logger.info(f"Loading adapter from {args.adapter}...")
    base_model = manager.base_model
    lora_model = PeftModel.from_pretrained(base_model, args.adapter)

    # Load evaluation data
    loader = DatasetLoader()
    data = loader.load_dataset(
        dataset_name=args.dataset,
        eval_size=args.num_samples
    )

    if args.compare:
        # Comparative evaluation
        logger.info("Running comparative evaluation...")
        comparator = ComparativeEvaluator(manager)

        # Compare models
        comparison = comparator.compare_models(
            base_model=base_model,
            lora_model=lora_model,
            eval_data=data['eval'],
            sample_size=args.num_samples
        )

        # Generate report
        variant_id = Path(args.adapter).name
        report_path = comparator.generate_report(comparison, variant_id)

        # Print results
        print("\n" + "="*60)
        print("COMPARATIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Adapter: {args.adapter}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {comparison['summary']['total_examples']}")
        print("\nAccuracy:")
        print(f"  Base Model:  {comparison['summary']['base_accuracy']:.2%}")
        print(f"  LoRA Model:  {comparison['summary']['lora_accuracy']:.2%}")
        print(f"  Improvement: {comparison['summary']['improvement']:.2%}")
        print("\nBreakdown:")
        print(f"  ✅ Improvements (Base wrong → LoRA right): {comparison['summary']['improvements_count']}")
        print(f"  ❌ Regressions (Base right → LoRA wrong): {comparison['summary']['regressions_count']}")
        print(f"  ✅✅ Both Correct: {comparison['summary']['both_correct_count']}")
        print(f"  ❌❌ Both Wrong: {comparison['summary']['both_wrong_count']}")
        print(f"\n📄 Full report saved to: {report_path}")
        print("="*60)
    else:
        # Simple evaluation
        evaluator = FitnessEvaluator(manager)
        metrics = evaluator.evaluate(
            model=lora_model,
            eval_data=data['eval'],
            variant_id="evaluation"
        )

        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Adapter: {args.adapter}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {metrics['total_evaluated']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Perplexity: {metrics['perplexity']:.2f}")
        print("="*60)


def list_datasets_command(args):
    """List available datasets"""
    loader = DatasetLoader()
    datasets = loader.list_available_datasets()

    print("\nAvailable Datasets:")
    print("-" * 40)
    for dataset in datasets:
        print(f"  - {dataset}")
    print("-" * 40)
    print(f"\nTotal: {len(datasets)} datasets available")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LoRALab Self-Supervised Evolution System"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Evolve command
    evolve_parser = subparsers.add_parser(
        'evolve',
        help='Run evolutionary optimization'
    )
    evolve_parser.add_argument(
        '--config',
        help='Configuration file path'
    )
    evolve_parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal settings'
    )
    evolve_parser.add_argument(
        '--resume',
        help='Resume from checkpoint file'
    )
    evolve_parser.add_argument(
        '--generations',
        type=int,
        help='Number of generations to run'
    )
    evolve_parser.add_argument(
        '--population',
        type=int,
        help='Population size per generation'
    )
    evolve_parser.add_argument(
        '--train-size',
        type=int,
        help='Number of training examples'
    )
    evolve_parser.add_argument(
        '--eval-size',
        type=int,
        help='Number of evaluation examples'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a specific adapter'
    )
    eval_parser.add_argument(
        '--adapter',
        required=True,
        help='Path to adapter directory'
    )
    eval_parser.add_argument(
        '--dataset',
        default='mmlu-pro',
        help='Dataset to evaluate on'
    )
    eval_parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to evaluate'
    )
    eval_parser.add_argument(
        '--config',
        help='Configuration file path'
    )
    eval_parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare base model vs LoRA adapter with detailed report'
    )

    # List datasets command
    list_parser = subparsers.add_parser(
        'list-datasets',
        help='List available datasets'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'evolve':
        evolve_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'list-datasets':
        list_datasets_command(args)


if __name__ == '__main__':
    main()