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
from .utils.cli_formatter import CLIFormatter, Fore, Style

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evolve_command(args):
    """Run evolutionary optimization"""
    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        CLIFormatter.print_info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        CLIFormatter.print_metric("CUDA memory", total_memory_gb, "GB")
    else:
        CLIFormatter.print_warning("CUDA not available - will use CPU (much slower)")

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

    # Apply pipeline test overrides if requested (minimal test for errors)
    if args.pipeline_test:
        CLIFormatter.print_header("PIPELINE TEST MODE", char="!")
        CLIFormatter.print_info("Running minimal pipeline test - checking for errors only")

        # Absolute minimum to test all components
        ss_config['evolution']['population_size'] = 3  # Min for crossover + mutation
        ss_config['evolution']['generations'] = 2      # Min to test evolution
        ss_config['evolution']['keep_top'] = 2         # Min survivors
        ss_config['dataset']['train_size'] = 10        # Min training examples
        ss_config['dataset']['eval_size'] = 5          # Min eval examples
        ss_config['training']['epochs_per_variant'] = 1
        ss_config['training']['batch_size'] = 2
        ss_config['training']['gradient_accumulation_steps'] = 1

        # Use only smallest ranks for speed
        ss_config['lora_search_space']['rank'] = [16, 32]

        CLIFormatter.print_info("Test configuration:")
        CLIFormatter.print_list_item("3 variants per generation (tests crossover + mutation)")
        CLIFormatter.print_list_item("2 generations (tests evolution)")
        CLIFormatter.print_list_item("10 training + 5 eval examples")
        CLIFormatter.print_list_item("Ranks limited to [16, 32]")

    # Apply quick test overrides if requested (slightly larger test)
    elif args.quick_test:
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
    CLIFormatter.print_header("FINAL RESULTS", char="*")

    final_results = {
        'Best Variant': best_variant.variant_id,
        'Accuracy': best_variant.eval_accuracy,
        'LoRA Rank': best_variant.rank,
        'Learning Rate': f"{float(best_variant.learning_rate):.0e}",
        'Output Directory': ss_config['output_dir']
    }

    CLIFormatter.print_summary_box("EVOLUTION SUMMARY", final_results, color=Fore.GREEN)


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
        CLIFormatter.print_header("COMPARATIVE EVALUATION RESULTS")

        # Basic info
        CLIFormatter.print_status("Adapter", args.adapter, Fore.CYAN, Fore.WHITE)
        CLIFormatter.print_status("Dataset", args.dataset, Fore.CYAN, Fore.WHITE)
        CLIFormatter.print_status("Samples", str(comparison['summary']['total_examples']), Fore.CYAN, Fore.WHITE)

        # Accuracy metrics
        CLIFormatter.print_subheader("Accuracy Metrics")
        CLIFormatter.print_metric(
            "Base Model",
            comparison['summary']['base_accuracy'],
            unit="%",
            good_threshold=0.7,
            bad_threshold=0.5
        )
        CLIFormatter.print_metric(
            "LoRA Model",
            comparison['summary']['lora_accuracy'],
            unit="%",
            good_threshold=0.7,
            bad_threshold=0.5
        )

        improvement = comparison['summary']['improvement']
        color = Fore.GREEN if improvement > 0 else Fore.RED
        CLIFormatter.print_status(
            "Improvement",
            f"{improvement:.2%}",
            Fore.CYAN,
            color
        )

        # Breakdown
        CLIFormatter.print_subheader("Performance Breakdown")
        breakdown_data = {
            'Improvements': f"{comparison['summary']['improvements_count']} (Base wrong -> LoRA right)",
            'Regressions': f"{comparison['summary']['regressions_count']} (Base right -> LoRA wrong)",
            'Both Correct': comparison['summary']['both_correct_count'],
            'Both Wrong': comparison['summary']['both_wrong_count']
        }

        for key, value in breakdown_data.items():
            CLIFormatter.print_list_item(f"{key}: {value}")

        CLIFormatter.print_success(f"Full report saved to: {report_path}")
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

    CLIFormatter.print_header("AVAILABLE DATASETS")

    # Dataset descriptions
    dataset_info = {
        'mmlu-pro': 'Multi-task language understanding benchmark',
        'gsm8k': 'Grade school math word problems',
        'squad': 'Stanford question answering dataset',
        'alpaca': 'Instruction-following dataset from Stanford',
        'dolly': 'Diverse instruction-following tasks by Databricks'
    }

    for i, dataset in enumerate(datasets, 1):
        desc = dataset_info.get(dataset, 'Custom dataset')
        CLIFormatter.print_list_item(
            f"{Fore.CYAN}{dataset}{Style.RESET_ALL} - {desc}",
            bullet=f"{i}."
        )

    CLIFormatter.print_info(f"Total: {len(datasets)} datasets available")


def analyze_command(args):
    """Analyze evolution history and create visualizations"""
    from .analysis.evolution_analyzer import EvolutionAnalyzer

    CLIFormatter.print_header("EVOLUTION ANALYSIS")

    # Check if output directory exists
    output_dir = Path(args.dir)
    if not output_dir.exists():
        CLIFormatter.print_error(f"Output directory not found: {output_dir}")
        sys.exit(1)

    # Check for history file
    history_file = output_dir / "history" / "evolution_history.json"
    if not history_file.exists():
        CLIFormatter.print_error("No evolution history found. Run evolution first.")
        sys.exit(1)

    CLIFormatter.print_info(f"Analyzing evolution history from: {output_dir}")

    # Run analysis
    analyzer = EvolutionAnalyzer(str(output_dir))

    if analyzer.load_history():
        analyzer.analyze()

        # Print summary of generated files
        analysis_dir = output_dir / "analysis" / "visualizations"
        CLIFormatter.print_header("ANALYSIS COMPLETE")

        files_created = [
            ('family_tree.png', 'Evolution family tree with lineage'),
            ('performance_timeline.png', 'Performance trends over generations'),
            ('hyperparameter_heatmap.png', 'Hyperparameter effectiveness'),
            ('survival_analysis.png', 'Survival rates by parameter'),
            ('mutation_effectiveness.png', 'Mutation impact analysis'),
            ('evolution_analysis_report.md', 'Comprehensive markdown report')
        ]

        CLIFormatter.print_subheader("Generated Files")
        for filename, description in files_created:
            file_path = analysis_dir / filename
            if file_path.exists():
                CLIFormatter.print_list_item(f"{Fore.GREEN}{filename}{Style.RESET_ALL}: {description}")

        CLIFormatter.print_success(f"All analysis files saved to: {analysis_dir}")
    else:
        CLIFormatter.print_error("Failed to load evolution history")


def list_runs_command(args):
    """List all evolution runs"""
    from .utils.output_manager import OutputManager

    CLIFormatter.print_header("EVOLUTION RUNS")

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        CLIFormatter.print_info(f"No runs found in {base_dir}")
        return

    manager = OutputManager(str(base_dir))
    runs = manager.list_runs()

    if not runs:
        CLIFormatter.print_info("No runs found")
        return

    CLIFormatter.print_info(f"Found {len(runs)} runs in {base_dir}:\n")

    for i, run_name in enumerate(runs, 1):
        run_path = base_dir / run_name

        # Get run info
        info_str = f"{Fore.CYAN}{run_name}{Style.RESET_ALL}"

        # Check for key files
        has_history = (run_path / "history" / "evolution_history.json").exists()
        has_best = (run_path / "models" / "best").exists()

        status_items = []
        if has_history:
            status_items.append(f"{Fore.GREEN}✓ History{Style.RESET_ALL}")
        if has_best:
            status_items.append(f"{Fore.GREEN}✓ Best Model{Style.RESET_ALL}")

        if status_items:
            info_str += f" [{', '.join(status_items)}]"

        CLIFormatter.print_list_item(info_str, bullet=f"{i}.")

    if args.cleanup:
        CLIFormatter.print_subheader(f"\nCleaning up old runs (keeping last {args.keep})")
        manager.cleanup_old_runs(keep_last=args.keep)
        CLIFormatter.print_success(f"Cleanup complete. Kept {args.keep} most recent runs.")


def print_welcome():
    """Print welcome banner"""
    CLIFormatter.print_header("LoRALab Evolution System v2.0", char="*")

    print(f"{Fore.CYAN}{'Self-Supervised LoRA Optimization'.center(CLIFormatter.get_terminal_width())}")
    print(f"{Fore.BLUE}{'No Teacher Models Required'.center(CLIFormatter.get_terminal_width())}{Style.RESET_ALL}\n")


def main():
    """Main CLI entry point"""
    # Show welcome banner only when no arguments provided
    if len(sys.argv) == 1:
        print_welcome()

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
        '--pipeline-test',
        action='store_true',
        help='Run minimal pipeline test to check for errors (not for quality)'
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

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze evolution history and create visualizations'
    )
    analyze_parser.add_argument(
        '--dir',
        required=True,
        help='Path to specific run directory (e.g., lora_runs/run_20250118_093000)'
    )

    # List runs command
    runs_parser = subparsers.add_parser(
        'list-runs',
        help='List all evolution runs'
    )
    runs_parser.add_argument(
        '--base-dir',
        default='lora_runs',
        help='Base directory containing runs (default: lora_runs)'
    )
    runs_parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old runs'
    )
    runs_parser.add_argument(
        '--keep',
        type=int,
        default=5,
        help='Number of runs to keep when cleaning up (default: 5)'
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
    elif args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'list-runs':
        list_runs_command(args)


if __name__ == '__main__':
    main()
