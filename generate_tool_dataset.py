"""Standalone script to generate tool calling dataset"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from loralab.benchmarks.tool_calling_dataset_generator import ToolCallDatasetGenerator

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("loralab/config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate tool calling dataset")
    parser.add_argument(
        "--model",
        help="Path to model for generation (optional - uses config.yaml if not provided)",
        default=None
    )
    parser.add_argument(
        "--use-challenger",
        action="store_true",
        help="Use the challenger model from config.yaml for generation"
    )
    parser.add_argument(
        "--use-solver",
        action="store_true",
        help="Use the solver model from config.yaml for generation"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for dataset (default: data/tool_calling_datasets/YYYYMMDD_HHMMSS.json)"
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only generate synthetic examples (no LLM generation)"
    )

    args = parser.parse_args()

    # Load config if needed
    model_path = args.model
    if not model_path and (args.use_challenger or args.use_solver):
        config = load_config()
        if config:
            if args.use_challenger:
                model_path = config.get('challenger', {}).get('model_path')
                print(f"Using challenger model from config: {model_path}")
            elif args.use_solver:
                # For solver, we'd need to handle the transformers model differently
                model_name = config.get('solver', {}).get('model_name')
                print(f"Note: Solver model {model_name} is a transformers model, using synthetic generation")
                model_path = None
        else:
            print("Warning: Could not load config.yaml")

    # Set up versioned output path if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        # Create timestamped filename in datasets directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        datasets_dir = Path("data/tool_calling_datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_path = datasets_dir / f"{timestamp}.json"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TOOL CALLING DATASET GENERATOR")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print(f"Dataset size: {args.size}")
    print(f"Mode: {'Synthetic only' if args.synthetic_only else 'Full generation'}")
    print()

    # Initialize generator
    if model_path and not args.synthetic_only:
        print(f"Using model: {model_path}")
        generator = ToolCallDatasetGenerator(model_path)
    else:
        print("Generating synthetic dataset (no model required)")
        # Use None for model path - we'll only use synthetic generation
        generator = ToolCallDatasetGenerator(None)

    # Generate dataset components
    print("Generating examples...")
    dataset = []

    # Calculate distribution
    per_category = args.size // 5

    print(f"  - Extraction examples: {per_category}")
    dataset.extend(generator.generate_extraction_examples(per_category))

    print(f"  - Math examples: {per_category}")
    dataset.extend(generator.generate_math_examples(per_category))

    print(f"  - Search examples: {per_category}")
    dataset.extend(generator.generate_search_examples(per_category))

    print(f"  - Event examples: {per_category}")
    dataset.extend(generator.generate_event_examples(per_category))

    print(f"  - Complex examples: {per_category}")
    dataset.extend(generator.generate_complex_examples(per_category))

    # Add LLM-generated examples if model is provided and not synthetic-only
    if model_path and not args.synthetic_only and args.size > 50:
        print(f"  - LLM-generated examples: {args.size // 10}")
        try:
            llm_examples = generator.generate_llm_created_examples(args.size // 10)
            dataset.extend(llm_examples)
        except Exception as e:
            print(f"    Warning: Could not generate LLM examples: {e}")
            print("    Continuing with synthetic examples only")

    # Save dataset
    print(f"\nSaving {len(dataset)} examples to {output_path}...")
    generator.save_dataset(dataset, str(output_path))

    # Generate statistics
    stats = {
        "total_examples": len(dataset),
        "generated_at": datetime.now().isoformat(),
        "by_category": {},
        "by_difficulty": {},
        "by_tool": {}
    }

    for example in dataset:
        cat = example.category
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        diff = example.difficulty
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        tool = example.expected_tool
        stats["by_tool"][tool] = stats["by_tool"].get(tool, 0) + 1

    # Save statistics
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  By category: {json.dumps(stats['by_category'], indent=4)}")
    print(f"  By difficulty: {json.dumps(stats['by_difficulty'], indent=4)}")
    print(f"  By tool: {json.dumps(stats['by_tool'], indent=4)}")

    print(f"\nDataset saved to: {output_path}")
    print(f"Statistics saved to: {stats_path}")

    # Create a symlink to latest dataset (platform-independent)
    if not args.output:  # Only if using auto-versioning
        latest_link = output_path.parent / "latest.json"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            # On Windows, create a copy instead of symlink for simplicity
            import shutil
            shutil.copy2(output_path, latest_link)
            print(f"Latest dataset link: {latest_link}")
        except Exception as e:
            print(f"Note: Could not create latest link: {e}")

    # Show sample examples
    print("\nSample examples from dataset:")
    import random
    samples = random.sample(dataset, min(3, len(dataset)))
    for i, example in enumerate(samples, 1):
        print(f"\nExample {i}:")
        print(f"  User: {example.user_input}")
        print(f"  Expected tool: {example.expected_tool}")
        print(f"  Expected params: {example.expected_parameters}")
        print(f"  Difficulty: {example.difficulty}")

if __name__ == "__main__":
    main()