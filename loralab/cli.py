"""
Command-line interface for LoRA evolution system.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from loralab.config.config_loader import ConfigLoader
from loralab.engine.lora_evolution import LoRAEvolution, load_evolution_checkpoint
from loralab.core.solver_client import SolverModel
from loralab.utils.monitor import EvolutionMonitor

logger = logging.getLogger(__name__)


def evolve_command(args):
    """Run evolution training."""
    print(" Starting LoRA evolution training...")

    # Load configuration
    config = ConfigLoader.load_config(args.config)

    # Override with command line arguments
    if args.output:
        config['output_dir'] = args.output
    if args.generations:
        config['evolution']['generations'] = args.generations
    if hasattr(args, 'use_direct') and args.use_direct:
        config['challenger']['use_direct'] = True
    if args.challenger_config:
        challenger_override = ConfigLoader.load_config(args.challenger_config)
        config['challenger'] = ConfigLoader.merge_configs(
            config['challenger'],
            challenger_override
        )
    if args.solver_config:
        solver_override = ConfigLoader.load_config(args.solver_config)
        config['solver'] = ConfigLoader.merge_configs(
            config['solver'],
            solver_override
        )

    # Create evolution instance based on mode
    # Force batched mode when using direct (can't have both models in memory)
    use_batched = (hasattr(args, 'batched') and args.batched) or \
                  (hasattr(args, 'use_direct') and args.use_direct)

    if use_batched:
        # Use memory-efficient batched mode
        if hasattr(args, 'use_direct') and args.use_direct:
            print(" Using DIRECT mode with automatic batching")
            print("   Models will alternate to save memory")
        else:
            print(" Using memory-efficient BATCHED mode")
            print("   Models will be loaded one at a time to save memory")
        from loralab.engine.batched_evolution import BatchedEvolution
        evolution = BatchedEvolution(config)
    else:
        # Use standard mode (both models in memory)
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            evolution = load_evolution_checkpoint(args.resume, config)
        else:
            evolution = LoRAEvolution(config)

    # Run evolution
    try:
        evolution.run()
        print("Evolution completed successfully!")
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"X Evolution failed: {e}")
        logger.exception("Evolution failed")
        sys.exit(1)


def test_command(args):
    """Test a trained adapter."""
    print("Testing LoRA adapter...")

    # Load adapter configuration
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"X Adapter not found: {adapter_path}")
        sys.exit(1)

    # Load metadata if available
    metadata_file = adapter_path.parent / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Adapter from generation {metadata.get('generation', 'unknown')}")
        print(f"Best score: {metadata.get('best_score', 'N/A'):.3f}")

    # Create solver model
    solver_config = {
        'model_name': args.model or 'Qwen/Qwen3-4B-Instruct-2507',
        'device': args.device or 'cuda',
        'lora_config': {
            'rank': 16,
            'alpha': 32
        }
    }

    print(f"Loading model: {solver_config['model_name']}")
    solver = SolverModel(solver_config)

    # Load adapter
    print(f"Loading adapter from: {adapter_path}")
    solver.load_adapter(str(adapter_path))

    # Test with input
    if args.input:
        # Read input file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"X Input file not found: {input_path}")
            sys.exit(1)

        with open(input_path, 'r') as f:
            input_text = f.read()

        print(f"Processing input: {input_path.name}")

        # Generate based on task type
        if args.task_type == 'code_documentation':
            prompt = f"Generate comprehensive documentation for this code:\n\n{input_text}\n\nDocumentation:"
        else:
            prompt = f"Task: {input_text}\n\nResponse:"

        # Generate response
        response = solver.generate(
            prompt,
            max_tokens=args.max_tokens or 512,
            temperature=args.temperature or 0.3
        )[0]

        print("\n" + "="*60)
        print("Generated Output:")
        print("="*60)
        print(response)

        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(response)
            print(f"\nOutput saved to: {output_path}")

    else:
        # Interactive mode
        print("\nInteractive mode (type 'quit' to exit)")
        while True:
            try:
                user_input = input("\nEnter prompt: ")
                if user_input.lower() == 'quit':
                    break

                response = solver.generate(
                    user_input,
                    max_tokens=256,
                    temperature=0.7
                )[0]

                print("\nResponse:", response)

            except KeyboardInterrupt:
                break

    print("\nTesting completed!")


def monitor_command(args):
    """Monitor evolution progress."""
    print("Monitoring evolution progress...")

    experiment_dir = Path(args.experiment)
    if not experiment_dir.exists():
        print(f"X Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Create monitor
    monitor = EvolutionMonitor(experiment_dir)

    if args.live:
        # Live monitoring mode
        print("Live monitoring mode (press Ctrl+C to stop)")
        monitor.live_monitor(refresh_interval=args.refresh or 5)
    else:
        # Show summary
        monitor.show_summary()

        # Generate plots if requested
        if args.plot:
            monitor.generate_plots()
            print(f"Plots saved to: {experiment_dir / 'plots'}")


def merge_command(args):
    """Merge LoRA adapter with base model."""
    print("Merging LoRA adapter with base model...")

    # Load configuration
    solver_config = {
        'model_name': args.model,
        'device': args.device or 'cuda',
        'lora_config': {
            'rank': 16,
            'alpha': 32
        }
    }

    # Load model with adapter
    print(f"Loading model: {args.model}")
    solver = SolverModel(solver_config)

    print(f"Loading adapter: {args.adapter}")
    solver.load_adapter(args.adapter)

    # Merge and save
    print("Merging LoRA weights with base model...")
    merged_model = solver.merge_and_unload()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    solver.tokenizer.save_pretrained(output_path)

    print("Model merged and saved successfully!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LoRALab - R-Zero inspired LoRA adapter generator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Run evolution training')
    evolve_parser.add_argument('--config', required=True, help='Configuration file (YAML/JSON)')
    evolve_parser.add_argument('--output', help='Output directory')
    evolve_parser.add_argument('--generations', type=int, help='Number of generations')
    evolve_parser.add_argument('--challenger-config', help='Override challenger configuration')
    evolve_parser.add_argument('--solver-config', help='Override solver configuration')
    evolve_parser.add_argument('--resume', help='Resume from checkpoint directory')
    evolve_parser.add_argument('--batched', action='store_true',
                              help='Use memory-efficient batched mode (loads one model at a time)')
    evolve_parser.add_argument('--use-direct', action='store_true',
                              help='Use direct llama-cpp-python instead of server for Challenger')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test a trained adapter')
    test_parser.add_argument('--adapter', required=True, help='Path to adapter directory')
    test_parser.add_argument('--model', help='Base model name/path')
    test_parser.add_argument('--input', help='Input file to process')
    test_parser.add_argument('--output', help='Output file to save results')
    test_parser.add_argument('--task-type', default='code_documentation', help='Task type')
    test_parser.add_argument('--max-tokens', type=int, help='Maximum tokens to generate')
    test_parser.add_argument('--temperature', type=float, help='Sampling temperature')
    test_parser.add_argument('--device', help='Device (cuda/cpu)')

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor evolution progress')
    monitor_parser.add_argument('--experiment', required=True, help='Experiment directory')
    monitor_parser.add_argument('--live', action='store_true', help='Live monitoring mode')
    monitor_parser.add_argument('--refresh', type=int, help='Refresh interval in seconds')
    monitor_parser.add_argument('--plot', action='store_true', help='Generate plots')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge adapter with base model')
    merge_parser.add_argument('--adapter', required=True, help='Path to adapter directory')
    merge_parser.add_argument('--model', required=True, help='Base model name/path')
    merge_parser.add_argument('--output', required=True, help='Output directory for merged model')
    merge_parser.add_argument('--device', help='Device (cuda/cpu)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'evolve':
        evolve_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'monitor':
        monitor_command(args)
    elif args.command == 'merge':
        merge_command(args)


if __name__ == '__main__':
    main()