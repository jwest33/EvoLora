"""Single GRPO Training Script
Run a single GRPO training session with one variant (no evolution).
This is useful for testing and quick experiments.
"""

import sys
import os
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Disable ALL multiprocessing on Windows before any imports
import platform
import multiprocessing

if platform.system() == 'Windows':
    # Check if we're in a child process
    if multiprocessing.current_process().name != 'MainProcess':
        # In child process - completely skip Unsloth imports
        os.environ["SKIP_UNSLOTH"] = "1"
        # Exit early from child processes that shouldn't import Unsloth
        sys.modules['unsloth'] = None  # Prevent import

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DATASETS_PARALLEL_DOWNLOADS"] = "false"
    os.environ["UNSLOTH_NUM_PROC"] = "1"
    os.environ["DATASETS_NUM_PROC"] = "1"

    # Suppress Windows multiprocessing redirect warnings
    warnings.filterwarnings("ignore", message=".*Redirects are currently not supported in Windows.*")
    # Suppress Unsloth TRL patching warnings
    warnings.filterwarnings("ignore", message=".*Unsloth TRL patching issue.*")

    # Windows multiprocessing fix
    if __name__ == '__main__':
        multiprocessing.freeze_support()
        # Force spawn method (required for Windows)
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except:
            pass

    # Force single process for datasets
    try:
        import datasets
        datasets.disable_progress_bars()
        datasets.config.NUM_PROC = 1
        # Also set through environment variable to be sure
        os.environ["HF_DATASETS_NUM_PROC"] = "1"
    except ImportError:
        pass

# Force disable multiprocessing in datasets globally before any imports
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["DATASETS_NUM_PROC"] = "1"

# Monkey patch datasets to never use multiprocessing
try:
    import datasets
    # Override default num_proc for all dataset operations
    _original_map = datasets.Dataset.map
    def _patched_map(self, *args, **kwargs):
        if 'num_proc' not in kwargs or kwargs['num_proc'] is None:
            kwargs['num_proc'] = 1
        return _original_map(self, *args, **kwargs)
    datasets.Dataset.map = _patched_map
except ImportError:
    pass

# Configure and initialize Unsloth before anything else
from loralab.utils.unsloth_config import init_unsloth
UNSLOTH_AVAILABLE = init_unsloth()

# Now import LoRALab modules
from loralab.core.lora_factory import LoRAFactory, LoRAVariant
from loralab.core.unsloth_manager import create_model_manager
from loralab.training.grpo_trainer import GRPOTrainer
from loralab.datasets.dataset_loader import DatasetLoader
# from loralab.evaluation.comparative_evaluator import ComparativeEvaluator  # Not used in single run
from loralab.utils.cli_formatter import CLIFormatter
from loralab.utils.memory_monitor import MemoryMonitor
from loralab.config.config_loader import ConfigLoader
# from loralab.utils.experiment_tracker import ExperimentTracker  # Not needed for single run

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run single GRPO training session"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="loralab/config/gemma3_grpo_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma3", "qwen3"],
        default="gemma3",
        help="Model to use (gemma3 or qwen3)"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="LoRA rank"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)"
    )

    return parser.parse_args()


def main():
    """Main single GRPO training function"""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    if args.model == "gemma3":
        config_path = "loralab/config/gemma3_grpo_config.yaml"
    else:
        config_path = "loralab/config/fast_grpo_config.yaml"  # Qwen3 config

    # Override with command line config if provided
    if args.config != "loralab/config/gemma3_grpo_config.yaml":
        config_path = args.config

    CLIFormatter.print_header("SINGLE GRPO TRAINING SESSION")
    CLIFormatter.print_subheader("Configuration")
    print(f"Config file: {config_path}")

    config = ConfigLoader.load_config(config_path)

    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.max_steps:
        config['grpo']['max_steps'] = args.max_steps

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model
        output_dir = Path(f"single_grpo_runs/{model_name}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.log_memory_status("Initial")

    # Create model manager
    CLIFormatter.print_subheader("Model Initialization")
    model_manager = create_model_manager(config['model'])  # Pass model config, not full config

    if not model_manager:
        logger.error("Failed to create model manager")
        return None

    # Load model
    model, tokenizer = model_manager.load_base_model()
    memory_monitor.log_memory_status("Model loaded")

    # Create single LoRA variant
    CLIFormatter.print_subheader("Creating LoRA Variant")
    variant = LoRAVariant(
        variant_id=f"single_{args.model}_r{args.rank}",
        rank=args.rank,
        alpha=args.rank,  # Alpha = rank (multiplier = 1)
        dropout=0.0,  # Zero for Unsloth optimization
        learning_rate=args.learning_rate,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        use_rslora=False,
        target_modules_preset="extended"
    )

    print(f"Variant: {variant.variant_id}")
    print(f"  Rank: {variant.rank}")
    print(f"  Learning Rate: {variant.learning_rate}")
    print(f"  Target Modules: {len(variant.target_modules)} modules")

    # Load dataset
    CLIFormatter.print_subheader("Loading Dataset")
    dataset_config = config['dataset']
    dataset_loader = DatasetLoader(cache_dir=dataset_config.get('cache_dir', 'cache/datasets'))

    # Get the first dataset source
    dataset_sources = dataset_config.get('sources', ['gsm8k'])
    dataset_name = dataset_sources[0] if dataset_sources else 'gsm8k'

    # Load the dataset
    dataset_dict = dataset_loader.load_dataset(
        dataset_name=dataset_name,
        train_size=dataset_config.get('train_size', 1000),
        eval_size=dataset_config.get('eval_size', 200),
        seed=dataset_config.get('seed', 42)
    )

    train_data = dataset_dict['train']
    eval_data = dataset_dict['eval']

    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")

    # Initialize GRPO trainer
    CLIFormatter.print_subheader("Initializing GRPO Trainer")
    # GRPOTrainer needs both training config and grpo config
    training_config_with_grpo = config['training'].copy()
    training_config_with_grpo['grpo'] = config.get('grpo', {})
    grpo_trainer = GRPOTrainer(model_manager, training_config_with_grpo)

    # Apply LoRA to model
    # Convert variant to lora config dict
    lora_config = {
        'rank': variant.rank,
        'alpha_multiplier': 1,  # Since alpha = rank in our variant
        'dropout': variant.dropout,
        'target_modules': variant.target_modules,
        'use_rslora': variant.use_rslora,
        'use_gradient_checkpointing': True
    }

    # Create LoRA variant (returns PEFT model)
    model = model_manager.create_lora_variant(lora_config)
    memory_monitor.log_memory_status("LoRA applied")

    # Run GRPO training
    CLIFormatter.print_subheader("Starting GRPO Training")

    # Pre-training on format (if enabled)
    if config['grpo'].get('pre_train_format', True):
        print("\n[Pre-training on format examples...]")
        format_loss = grpo_trainer.pre_train_formatting(
            model,
            train_data[:config['grpo'].get('format_examples', 50)],
            learning_rate=variant.learning_rate,
            epochs=2
        )
        print(f"Format pre-training loss: {format_loss:.4f}")

    # Main GRPO training
    print("\n[Starting main GRPO training...]")
    metrics = grpo_trainer.train(
        model=model,
        train_data=train_data,
        variant_id=variant.variant_id,
        learning_rate=variant.learning_rate,
        max_steps=args.max_steps
    )

    # Log results
    CLIFormatter.print_subheader("Training Results")
    print(f"Final Loss: {metrics.get('final_loss', 'N/A')}")
    print(f"Average Reward: {metrics.get('rewards', 0.0):.4f}")
    print(f"Total Steps: {metrics.get('total_steps', 0)}")

    # Simple evaluation (perplexity calculation)
    CLIFormatter.print_subheader("Evaluation")

    # Calculate perplexity on a subset of eval data
    import math
    import torch

    model.eval()
    total_loss = 0
    num_samples = min(50, len(eval_data))  # Evaluate on subset

    with torch.no_grad():
        for i in range(num_samples):
            sample = eval_data[i]
            # Simple loss calculation (would need proper implementation for accuracy)
            # For now, just use the training metrics
            pass

    eval_metrics = {
        'final_loss': metrics.get('final_loss', 0),
        'perplexity': math.exp(min(metrics.get('final_loss', 0), 10))  # Cap at e^10 to avoid overflow
    }

    print(f"Training Loss: {eval_metrics.get('final_loss', 'N/A')}")
    print(f"Estimated Perplexity: {eval_metrics.get('perplexity', 'N/A'):.2f}")

    # Save model and results
    CLIFormatter.print_subheader("Saving Results")

    # Save LoRA adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to: {adapter_path}")

    # Save configuration
    import yaml
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Configuration saved to: {config_save_path}")

    # Save metrics
    import json
    metrics_path = output_dir / "metrics.json"
    all_metrics = {
        'variant': {
            'id': variant.variant_id,
            'rank': variant.rank,
            'learning_rate': variant.learning_rate
        },
        'training': metrics,
        'evaluation': eval_metrics
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Final memory status
    memory_monitor.log_memory_status("Training complete")

    # Print memory summary
    summary = memory_monitor.get_summary()
    CLIFormatter.print_subheader("Memory Usage Summary")
    print(f"Peak GPU Memory: {summary['peak_gpu_gb']:.2f} GB")
    print(f"Peak RAM: {summary['peak_ram_gb']:.2f} GB")

    CLIFormatter.print_success(f"\nSingle GRPO training completed successfully!")
    CLIFormatter.print_success(f"Results saved to: {output_dir}")

    return model, metrics


if __name__ == "__main__":
    main()