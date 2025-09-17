"""Tool Calling Experiment Runner"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from tool_calling_dataset_generator import ToolCallDatasetGenerator
from tool_calling_evaluator import ToolCallEvaluator

# Import dataset utilities
try:
    from dataset_utils import load_dataset, get_latest_dataset, list_datasets
except ImportError:
    from .dataset_utils import load_dataset, get_latest_dataset, list_datasets

class ToolCallingExperiment:
    """Orchestrates tool calling experiments"""

    def __init__(self, large_model_path: str, base_model_path: str, output_dir: str = "experiments/tool_calling"):
        """
        Initialize experiment with model paths

        Args:
            large_model_path: Path to larger model for dataset generation
            base_model_path: Path to base model to evaluate
            output_dir: Directory to save results
        """
        self.large_model_path = large_model_path
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped experiment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"experiment_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, dataset_size: int = 100, variations: List[str] = None,
                       use_existing_dataset: bool = True):
        """
        Run complete experiment pipeline

        Args:
            dataset_size: Number of examples to generate
            variations: List of prompt variations to test
            use_existing_dataset: If True, use latest existing dataset if available
        """
        print("=" * 60)
        print("TOOL CALLING EXPERIMENT")
        print("=" * 60)
        print(f"Output directory: {self.experiment_dir}")
        print()

        # Step 1: Generate or use existing dataset
        if use_existing_dataset:
            dataset_path = get_latest_dataset()
            if dataset_path:
                print(f"Step 1: Using existing dataset: {dataset_path}")
                data = load_dataset(dataset_path)
                print(f"  Dataset contains {len(data['examples'])} examples")
            else:
                print("Step 1: No existing dataset found, generating new one...")
                dataset_path = self._generate_dataset(dataset_size)
        else:
            print("Step 1: Generating new dataset with larger model...")
            dataset_path = self._generate_dataset(dataset_size)

        # Step 2: Baseline evaluation
        print("\nStep 2: Running baseline evaluation...")
        baseline_metrics = self._evaluate_baseline(dataset_path)

        # Step 3: Test variations (if provided)
        variation_results = {}
        if variations:
            print(f"\nStep 3: Testing {len(variations)} prompt variations...")
            variation_results = self._test_variations(dataset_path, variations)

        # Step 4: Generate comprehensive report
        print("\nStep 4: Generating comprehensive report...")
        self._generate_comprehensive_report(baseline_metrics, variation_results)

        print("\nExperiment complete!")
        print(f"Results saved in: {self.experiment_dir}")

    def _generate_dataset(self, size: int) -> str:
        """Generate tool calling dataset"""
        generator = ToolCallDatasetGenerator(self.large_model_path)

        # Generate diverse dataset
        dataset = generator.generate_dataset(size)

        # Add LLM-generated examples for more diversity
        if size > 50:
            llm_examples = generator.generate_llm_created_examples(size // 5)
            dataset.extend(llm_examples)

        # Save dataset
        dataset_path = self.experiment_dir / "dataset.json"
        generator.save_dataset(dataset, str(dataset_path))

        # Save dataset statistics
        self._save_dataset_stats(dataset)

        return str(dataset_path)

    def _save_dataset_stats(self, dataset: List):
        """Save statistics about the generated dataset"""
        stats = {
            "total_examples": len(dataset),
            "by_category": {},
            "by_difficulty": {},
            "by_tool": {}
        }

        for example in dataset:
            # Count by category
            cat = example.category
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Count by difficulty
            diff = example.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # Count by tool
            tool = example.expected_tool
            stats["by_tool"][tool] = stats["by_tool"].get(tool, 0) + 1

        stats_path = self.experiment_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Dataset statistics:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Categories: {stats['by_category']}")
        print(f"  Difficulties: {stats['by_difficulty']}")

    def _evaluate_baseline(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate base model on dataset"""
        evaluator = ToolCallEvaluator(self.base_model_path)

        # Run evaluation
        metrics = evaluator.evaluate_dataset(dataset_path)

        # Save metrics
        metrics_path = self.experiment_dir / "baseline_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Generate report
        report_path = self.experiment_dir / "baseline_report.txt"
        evaluator.generate_report(metrics, str(report_path))

        return metrics

    def _test_variations(self, dataset_path: str, variations: List[str]) -> Dict[str, Dict]:
        """Test different prompt variations"""
        results = {}

        for i, system_prompt in enumerate(variations):
            print(f"  Testing variation {i+1}/{len(variations)}...")

            # Create modified evaluator with custom prompt
            evaluator = CustomPromptEvaluator(self.base_model_path, system_prompt)

            # Evaluate
            metrics = evaluator.evaluate_dataset(dataset_path)

            # Save results
            variation_dir = self.experiment_dir / f"variation_{i+1}"
            variation_dir.mkdir(exist_ok=True)

            with open(variation_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

            with open(variation_dir / "prompt.txt", 'w') as f:
                f.write(system_prompt)

            results[f"variation_{i+1}"] = metrics

        return results

    def _generate_comprehensive_report(self, baseline: Dict, variations: Dict):
        """Generate comprehensive experiment report"""
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE TOOL CALLING EXPERIMENT REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Base Model: {self.base_model_path}")
        report.append("")

        # Baseline results
        report.append("BASELINE RESULTS:")
        report.append(f"  Tool Accuracy: {baseline['tool_accuracy']:.2%}")
        report.append(f"  Parameter Accuracy: {baseline['parameter_accuracy']:.2%}")
        report.append(f"  Full Accuracy: {baseline['full_accuracy']:.2%}")
        report.append(f"  Avg Response Time: {baseline['avg_response_time']:.3f}s")
        report.append("")

        # Variation comparisons
        if variations:
            report.append("PROMPT VARIATION RESULTS:")
            for name, metrics in variations.items():
                report.append(f"  {name}:")
                report.append(f"    Tool Accuracy: {metrics['tool_accuracy']:.2%} ({self._diff(metrics['tool_accuracy'], baseline['tool_accuracy'])})")
                report.append(f"    Parameter Accuracy: {metrics['parameter_accuracy']:.2%} ({self._diff(metrics['parameter_accuracy'], baseline['parameter_accuracy'])})")
                report.append(f"    Full Accuracy: {metrics['full_accuracy']:.2%} ({self._diff(metrics['full_accuracy'], baseline['full_accuracy'])})")
            report.append("")

        # Key findings
        report.append("KEY FINDINGS:")
        if variations:
            best_variation = max(variations.items(), key=lambda x: x[1]['full_accuracy'])
            if best_variation[1]['full_accuracy'] > baseline['full_accuracy']:
                improvement = (best_variation[1]['full_accuracy'] - baseline['full_accuracy']) * 100
                report.append(f"  Best performing variation: {best_variation[0]}")
                report.append(f"  Improvement over baseline: {improvement:.1f} percentage points")
            else:
                report.append("  Baseline outperformed all variations")

        report_text = "\n".join(report)

        # Save report
        report_path = self.experiment_dir / "comprehensive_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        print("\n" + report_text)

    def _diff(self, new_val: float, base_val: float) -> str:
        """Format difference from baseline"""
        diff = (new_val - base_val) * 100
        if diff > 0:
            return f"+{diff:.1f}%"
        else:
            return f"{diff:.1f}%"


class CustomPromptEvaluator(ToolCallEvaluator):
    """Evaluator with custom system prompt"""

    def __init__(self, model_path: str, system_prompt: str):
        super().__init__(model_path)
        self.system_prompt = system_prompt

    def evaluate_example(self, example: Dict, tools: List[Dict]) -> Any:
        """Override to use custom prompt"""
        # Temporarily modify the messages
        original_method = super().evaluate_example

        # Create modified example with custom system prompt
        modified_example = example.copy()

        # Call parent method with modification
        import time
        start_time = time.time()

        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": example["user_input"]
                }
            ]

            response = self.llm.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_time = time.time() - start_time

            predicted_tool, predicted_params = self._parse_response(response)

            correct_tool = predicted_tool == example["expected_tool"]
            correct_params = self._compare_parameters(
                example["expected_parameters"],
                predicted_params
            )

            from tool_calling_evaluator import EvaluationResult
            return EvaluationResult(
                example_id=example.get("id", 0),
                correct_tool=correct_tool,
                correct_params=correct_params,
                response_time=response_time,
                expected_tool=example["expected_tool"],
                predicted_tool=predicted_tool,
                expected_params=example["expected_parameters"],
                predicted_params=predicted_params
            )

        except Exception as e:
            from tool_calling_evaluator import EvaluationResult
            return EvaluationResult(
                example_id=example.get("id", 0),
                correct_tool=False,
                correct_params=False,
                response_time=time.time() - start_time,
                expected_tool=example["expected_tool"],
                predicted_tool="",
                expected_params=example["expected_parameters"],
                predicted_params={},
                error=str(e)
            )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run tool calling experiments")
    parser.add_argument("--large-model", help="Path to large model for generation (required if generating new dataset)")
    parser.add_argument("--base-model", required=True, help="Path to base model to evaluate")
    parser.add_argument("--dataset-size", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--output", default="experiments/tool_calling", help="Output directory")
    parser.add_argument("--generate-new", action="store_true", help="Force generation of new dataset")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")

    args = parser.parse_args()

    # Handle listing datasets
    if args.list_datasets:
        print("Available datasets:")
        datasets = list_datasets()
        if not datasets:
            print("  No datasets found")
        else:
            for ds in datasets:
                print(f"  {ds['name']}: {ds['size']} examples")
        return

    # Check if we need the large model
    if args.generate_new and not args.large_model:
        parser.error("--large-model is required when using --generate-new")

    # Define prompt variations to test
    variations = [
        "You are a helpful assistant. Always use the most appropriate function when needed.",
        "You are an AI assistant specialized in function calling. Analyze user requests carefully and select the correct function with accurate parameters.",
        "Assistant: I help users by calling appropriate functions. I carefully parse requests to extract the right information."
    ]

    # Run experiment
    experiment = ToolCallingExperiment(
        large_model_path=args.large_model or "dummy",  # Won't be used if dataset exists
        base_model_path=args.base_model,
        output_dir=args.output
    )

    experiment.run_experiment(
        dataset_size=args.dataset_size,
        variations=variations,
        use_existing_dataset=not args.generate_new
    )


if __name__ == "__main__":
    main()