"""Tool Calling Evaluation Framework"""

import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from llama_cpp import Llama
from pathlib import Path

# Import dataset utilities
try:
    from dataset_utils import load_dataset, get_latest_dataset
except ImportError:
    from .dataset_utils import load_dataset, get_latest_dataset

@dataclass
class EvaluationResult:
    """Results from evaluating a single example"""
    example_id: int
    correct_tool: bool
    correct_params: bool
    response_time: float
    expected_tool: str
    predicted_tool: str
    expected_params: Dict[str, Any]
    predicted_params: Dict[str, Any]
    error: str = None

class ToolCallEvaluator:
    """Evaluates base model's tool calling capabilities"""

    def __init__(self, model_path: str):
        """Initialize evaluator with base model"""
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            chat_format="chatml-function-calling",
            n_ctx=2048,
            n_gpu_layers=-1  # Use GPU if available
        )

    def evaluate_example(self, example: Dict, tools: List[Dict]) -> EvaluationResult:
        """Evaluate a single tool calling example"""
        start_time = time.time()

        try:
            # Create the message
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Call functions with appropriate input when necessary."
                },
                {
                    "role": "user",
                    "content": example["user_input"]
                }
            ]

            # Get model response
            response = self.llm.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_time = time.time() - start_time

            # Parse response
            predicted_tool, predicted_params = self._parse_response(response)

            # Evaluate correctness
            correct_tool = predicted_tool == example["expected_tool"]
            correct_params = self._compare_parameters(
                example["expected_parameters"],
                predicted_params
            )

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

    def _parse_response(self, response: Dict) -> Tuple[str, Dict]:
        """Parse tool name and parameters from model response"""
        if "choices" not in response or len(response["choices"]) == 0:
            return "", {}

        choice = response["choices"][0]
        message = choice.get("message", {})

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("function", {}).get("name", "")
            try:
                params = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            except:
                params = {}
            return tool_name, params

        # Check for function call (older format)
        function_call = message.get("function_call")
        if function_call:
            tool_name = function_call.get("name", "")
            try:
                params = json.loads(function_call.get("arguments", "{}"))
            except:
                params = {}
            return tool_name, params

        return "", {}

    def _compare_parameters(self, expected: Dict, predicted: Dict) -> bool:
        """Compare expected and predicted parameters"""
        if not expected and not predicted:
            return True

        # Check required keys
        for key in expected:
            if key not in predicted:
                return False
            if expected[key] != predicted[key]:
                # Allow some flexibility for numeric values
                if isinstance(expected[key], (int, float)) and isinstance(predicted[key], (int, float)):
                    if abs(expected[key] - predicted[key]) < 0.001:
                        continue
                return False

        return True

    def evaluate_dataset(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset.

        Args:
            dataset_path: Path to dataset file, or None to use latest

        Returns:
            Dictionary of evaluation metrics
        """
        # Load dataset (auto-detects latest if not specified)
        if dataset_path is None:
            dataset_path = get_latest_dataset()
            if dataset_path is None:
                raise FileNotFoundError("No datasets found")

        data = load_dataset(dataset_path)
        self.dataset_path = dataset_path  # Store for reporting

        tools = data["tools"]
        examples = data["examples"]
        results = []

        print(f"Evaluating {len(examples)} examples...")

        for i, example in enumerate(examples):
            example["id"] = i
            result = self.evaluate_example(example, tools)
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(examples)} examples")

        return self._compute_metrics(results)

    def _compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute evaluation metrics"""
        total = len(results)
        correct_tools = sum(r.correct_tool for r in results)
        correct_params = sum(r.correct_params for r in results)
        fully_correct = sum(r.correct_tool and r.correct_params for r in results)

        # Group by difficulty and category
        by_difficulty = {}
        by_category = {}

        for r in results:
            # Note: Would need to track difficulty/category in results
            pass

        metrics = {
            "total_examples": total,
            "tool_accuracy": correct_tools / total if total > 0 else 0,
            "parameter_accuracy": correct_params / total if total > 0 else 0,
            "full_accuracy": fully_correct / total if total > 0 else 0,
            "avg_response_time": sum(r.response_time for r in results) / total if total > 0 else 0,
            "errors": sum(1 for r in results if r.error is not None)
        }

        # Confusion matrix for tools
        tool_confusion = {}
        for r in results:
            if r.expected_tool not in tool_confusion:
                tool_confusion[r.expected_tool] = {}
            if r.predicted_tool not in tool_confusion[r.expected_tool]:
                tool_confusion[r.expected_tool][r.predicted_tool] = 0
            tool_confusion[r.expected_tool][r.predicted_tool] += 1

        metrics["tool_confusion"] = tool_confusion

        return metrics

    def generate_report(self, metrics: Dict[str, Any], output_path: str):
        """Generate detailed evaluation report"""
        from datetime import datetime
        from pathlib import Path

        report = []
        report.append("=" * 60)
        report.append("TOOL CALLING EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Metadata section
        report.append("EVALUATION METADATA:")
        report.append(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Model: {Path(self.model_path).name}")
        report.append(f"  Model Path: {self.model_path}")
        report.append(f"  Dataset: {Path(self.dataset_path).name if hasattr(self, 'dataset_path') else 'Unknown'}")
        report.append(f"  Dataset Path: {self.dataset_path if hasattr(self, 'dataset_path') else 'Unknown'}")
        report.append("")

        report.append("OVERALL METRICS:")
        report.append(f"  Total Examples: {metrics['total_examples']}")
        report.append(f"  Tool Selection Accuracy: {metrics['tool_accuracy']:.2%}")
        report.append(f"  Parameter Extraction Accuracy: {metrics['parameter_accuracy']:.2%}")
        report.append(f"  Full Task Accuracy: {metrics['full_accuracy']:.2%}")
        report.append(f"  Average Response Time: {metrics['avg_response_time']:.3f}s")
        report.append(f"  Errors: {metrics['errors']}")
        report.append("")

        report.append("TOOL CONFUSION MATRIX:")
        confusion = metrics.get("tool_confusion", {})
        for expected, predictions in confusion.items():
            report.append(f"  {expected}:")
            for predicted, count in predictions.items():
                report.append(f"    -> {predicted}: {count}")
        report.append("")

        # Add breakdown by difficulty if available
        report.append("PERFORMANCE BY CATEGORY:")
        report.append("  (Analysis of performance by category would go here)")
        report.append("")

        report_text = "\n".join(report)

        # Ensure parent directory exists
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report_text)

        # Don't print full report, just summary with metadata
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"  Model: {Path(self.model_path).name}")
        print(f"  Dataset: {Path(self.dataset_path).name if hasattr(self, 'dataset_path') else 'Unknown'}")
        print(f"  Tool Accuracy: {metrics['tool_accuracy']:.1%}")
        print(f"  Parameter Accuracy: {metrics['parameter_accuracy']:.1%}")
        print(f"  Full Accuracy: {metrics['full_accuracy']:.1%}")

        return report_text