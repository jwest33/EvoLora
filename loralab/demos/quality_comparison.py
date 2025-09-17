"""
Consolidated quality comparison demos for LoRA adapters.
Shows the difference between base model and LoRA-enhanced outputs.
"""

import torch
from pathlib import Path
from typing import List, Tuple, Optional
import json
from loralab.core.solver_client import SolverModel


class QualityComparison:
    """Compare base model vs LoRA-enhanced model outputs."""

    def __init__(self, adapter_path: Optional[str] = None):
        """
        Initialize comparison with optional adapter.

        Args:
            adapter_path: Path to LoRA adapter (optional, will auto-find if None)
        """
        self.adapter_path = adapter_path or self._find_best_adapter()

        # Load configuration
        self.config = self._load_config()

        # Initialize solver model
        print("Loading base model...")
        self.solver = SolverModel(self.config['solver'])

        if self.adapter_path:
            print(f"Loading LoRA adapter from {self.adapter_path}...")
            self.solver.load_adapter(self.adapter_path)

    def _find_best_adapter(self) -> Optional[str]:
        """Find the best adapter from experiments."""
        experiments = Path("experiments")
        if not experiments.exists():
            return None

        # Look for best checkpoints
        adapters = list(experiments.glob("*/best_checkpoint/adapter"))
        if adapters:
            # Return the most recent one
            return str(adapters[-1])
        return None

    def _load_config(self) -> dict:
        """Load configuration from default location."""
        import yaml
        config_path = Path("loralab/configs/documentation.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_test_samples(self) -> List[str]:
        """Get sample code for testing."""
        samples = [
            # Simple function
            '''def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    if count == 0:
        return 0
    return total / count''',

            # Complex function with edge cases
            '''def merge_sorted_lists(list1, list2, reverse=False):
    result = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if (list1[i] <= list2[j]) != reverse:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1

    result.extend(list1[i:])
    result.extend(list2[j:])
    return result''',

            # Recursive function
            '''def find_path(graph, start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return path

    if start not in graph:
        return None

    for node in graph[start]:
        if node not in path:
            new_path = find_path(graph, node, end, path)
            if new_path:
                return new_path

    return None''',

            # Generator function
            '''def fibonacci_generator(limit):
    a, b = 0, 1
    count = 0

    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

    return count''',

            # Context manager
            '''class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if exc_type:
            print(f"Error occurred: {exc_val}")
        return False'''
        ]
        return samples

    def generate_documentation(self, code: str, use_adapter: bool = True) -> str:
        """
        Generate documentation for code.

        Args:
            code: Python code to document
            use_adapter: Whether to use LoRA adapter

        Returns:
            Generated documentation
        """
        prompt = f'''Generate comprehensive documentation for this Python code:

{code}

Provide a detailed docstring including:
- Description of what the function/class does
- Parameters with types and descriptions
- Return value description
- Any exceptions that might be raised
- Example usage

Documentation:'''

        # Set adapter if needed
        if use_adapter and self.adapter_path:
            self.solver.set_adapter("default")
        elif not use_adapter and self.adapter_path:
            # Temporarily disable adapter
            self.solver.model.disable_adapters()

        # Generate documentation
        docs = self.solver.generate(
            prompt,
            max_tokens=512,
            temperature=0.3,
            num_return=1
        )[0]

        # Re-enable adapter if it was disabled
        if not use_adapter and self.adapter_path:
            self.solver.model.enable_adapters()

        return docs

    def quick_comparison(self):
        """Run quick visual comparison with 3 examples."""
        print("\n" + "=" * 60)
        print("QUICK QUALITY COMPARISON")
        print("=" * 60)

        samples = self.get_test_samples()[:3]

        for i, code in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"EXAMPLE {i}")
            print(f"{'='*60}")
            print("\nOriginal Code:")
            print("-" * 40)
            print(code)

            if self.adapter_path:
                # Generate with base model
                print("\n\nBase Model Output:")
                print("-" * 40)
                base_doc = self.generate_documentation(code, use_adapter=False)
                print(base_doc)

                # Generate with LoRA
                print("\n\nLoRA-Enhanced Output:")
                print("-" * 40)
                lora_doc = self.generate_documentation(code, use_adapter=True)
                print(lora_doc)
            else:
                print("\n\nModel Output:")
                print("-" * 40)
                doc = self.generate_documentation(code, use_adapter=False)
                print(doc)
                print("\n(No LoRA adapter found for comparison)")

            input("\nPress Enter to continue...")

    def full_analysis(self):
        """Run full quality analysis with scoring."""
        print("\n" + "=" * 60)
        print("FULL QUALITY ANALYSIS")
        print("=" * 60)

        samples = self.get_test_samples()

        results = {
            'base_scores': [],
            'lora_scores': [],
            'improvements': []
        }

        for i, code in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"ANALYZING EXAMPLE {i}/5")
            print(f"{'='*60}")

            if self.adapter_path:
                # Generate with both
                base_doc = self.generate_documentation(code, use_adapter=False)
                lora_doc = self.generate_documentation(code, use_adapter=True)

                # Score outputs (simple length and keyword analysis)
                base_score = self._score_documentation(base_doc, code)
                lora_score = self._score_documentation(lora_doc, code)

                results['base_scores'].append(base_score)
                results['lora_scores'].append(lora_score)
                results['improvements'].append(lora_score - base_score)

                print(f"Base Model Score: {base_score:.2f}")
                print(f"LoRA Model Score: {lora_score:.2f}")
                print(f"Improvement: {(lora_score - base_score):.2f} ({((lora_score/base_score - 1) * 100):.1f}%)")
            else:
                doc = self.generate_documentation(code, use_adapter=False)
                score = self._score_documentation(doc, code)
                results['base_scores'].append(score)
                print(f"Model Score: {score:.2f}")
                print("(No LoRA adapter found for comparison)")

        # Summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")

        if self.adapter_path and results['lora_scores']:
            avg_base = sum(results['base_scores']) / len(results['base_scores'])
            avg_lora = sum(results['lora_scores']) / len(results['lora_scores'])
            avg_improvement = sum(results['improvements']) / len(results['improvements'])

            print(f"\nAverage Scores:")
            print(f"  Base Model:  {avg_base:.2f}")
            print(f"  LoRA Model:  {avg_lora:.2f}")
            print(f"  Improvement: {avg_improvement:.2f} ({(avg_improvement/avg_base * 100):.1f}%)")

            print(f"\nBest Improvement: {max(results['improvements']):.2f}")
            print(f"Worst Improvement: {min(results['improvements']):.2f}")
        else:
            avg_base = sum(results['base_scores']) / len(results['base_scores'])
            print(f"\nAverage Score: {avg_base:.2f}")

    def _score_documentation(self, doc: str, code: str) -> float:
        """
        Simple scoring function for documentation quality.

        Args:
            doc: Generated documentation
            code: Original code

        Returns:
            Quality score
        """
        score = 0.0

        # Length score (longer is generally better, up to a point)
        doc_length = len(doc.split())
        if doc_length > 50:
            score += min(doc_length / 100, 2.0)

        # Keyword presence
        keywords = ['Parameters', 'Args', 'Returns', 'Example', 'Raises', 'Note']
        for keyword in keywords:
            if keyword in doc:
                score += 0.5

        # Structure indicators
        if '"""' in doc or "'''" in doc:
            score += 0.5
        if '\n    ' in doc:  # Indentation
            score += 0.5

        # Function/class name mentioned
        if 'def ' in code:
            func_name = code.split('def ')[1].split('(')[0]
            if func_name in doc:
                score += 1.0

        # Type hints mentioned
        if 'type' in doc.lower() or '->' in doc:
            score += 0.5

        return min(score, 10.0)  # Cap at 10