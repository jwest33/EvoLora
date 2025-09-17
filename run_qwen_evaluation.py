"""Evaluate Qwen models with proper tool calling format"""

import argparse
from pathlib import Path
from datetime import datetime
from loralab.benchmarks.qwen_tool_evaluator import QwenToolEvaluator
from loralab.benchmarks.dataset_utils import get_latest_dataset, list_datasets

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen model on tool calling dataset")
    parser.add_argument(
        "--model",
        help="Path to Qwen model (defaults to challenger from config.yaml)"
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset (defaults to latest)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for report (defaults to reports/qwen_evaluation_TIMESTAMP.txt)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    # List datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        datasets = list_datasets()
        if not datasets:
            print("  No datasets found")
        else:
            for ds in datasets:
                print(f"  {ds['name']}: {ds['size']} examples")
        return

    # Get model path
    model_path = args.model
    if not model_path:
        # Try to load from config
        try:
            import yaml
            config_path = Path("loralab/config/config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_path = config.get('solver', {}).get('gguf_model_path')
                if model_path:
                    print(f"Using solver model from config: {model_path}")
        except:
            pass

        if not model_path:
            print("Error: No model specified")
            print("Options:")
            print("  1. Specify a model: python run_qwen_evaluation.py --model path/to/model.gguf")
            print("  2. Set solver model in loralab/config/config.yaml")
            return

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Initialize evaluator
    print(f"Loading Qwen model: {model_path}")
    evaluator = QwenToolEvaluator(model_path)

    # Get dataset
    dataset_path = args.dataset
    if not dataset_path:
        dataset_path = get_latest_dataset()
        if not dataset_path:
            print("Error: No datasets found. Generate one first with generate_tool_dataset.py")
            return
        print(f"Using latest dataset: {dataset_path}")

    # Run evaluation
    print("\nStarting Qwen-formatted evaluation...")
    metrics = evaluator.evaluate_dataset(dataset_path)

    # Set up output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / f"qwen_evaluation_{timestamp}.txt"

    # Generate report
    print(f"\nGenerating report...")
    evaluator.generate_report(metrics, str(output_path))
    print(f"Report saved to {output_path}")

    # Show detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")

    # Show confusion matrix
    if 'tool_confusion' in metrics:
        print("\nTool Selection Breakdown:")
        confusion = metrics['tool_confusion']
        for expected, predictions in confusion.items():
            if expected:  # Skip empty tool names
                total = sum(predictions.values())
                correct = predictions.get(expected, 0)
                print(f"  {expected}: {correct}/{total} ({100*correct/total:.1f}% correct)")

    print(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()
