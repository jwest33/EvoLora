"""Comparative evaluator for base model vs LoRA adapter

Generates human-readable reports showing improvement areas.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)


class ComparativeEvaluator:
    """Evaluates and compares base model vs LoRA adapter performance"""

    def __init__(self, model_manager, output_dir: str = "evaluation_reports"):
        """Initialize comparative evaluator

        Args:
            model_manager: ModelManager instance
            output_dir: Directory to save reports
        """
        self.model_manager = model_manager
        self.tokenizer = model_manager.get_tokenizer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def evaluate_single(self, model: Any, example: Dict, max_length: int = 50) -> Dict:
        """Evaluate a single example

        Args:
            model: Model to evaluate
            example: Example with 'question' and 'answer' fields
            max_length: Maximum generation length

        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        device = next(model.parameters()).device

        question = example.get('question', '')
        true_answer = example.get('answer', '').strip()

        # Format prompt
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated.replace(prompt, "").strip()

        # Check correctness
        is_correct = self._check_answer(true_answer, generated_answer)

        return {
            'question': question,
            'true_answer': true_answer,
            'generated_answer': generated_answer,
            'is_correct': is_correct,
            'prompt': prompt
        }

    def _check_answer(self, true_answer: str, generated_answer: str) -> bool:
        """Check if generated answer is correct

        Args:
            true_answer: Ground truth answer
            generated_answer: Model's answer

        Returns:
            True if correct, False otherwise
        """
        true_lower = true_answer.lower().strip()
        gen_lower = generated_answer.lower().strip()

        # Direct match
        if true_lower in gen_lower:
            return True

        # Multiple choice check
        if len(true_lower) == 1 and true_lower in 'abcde':
            for char in gen_lower:
                if char in 'abcde':
                    return char == true_lower

        # Word overlap check
        true_words = set(true_lower.split())
        gen_words = set(gen_lower.split())
        if len(true_words) > 0:
            overlap = len(true_words.intersection(gen_words)) / len(true_words)
            if overlap >= 0.8:
                return True

        return False

    def compare_models(self,
                      base_model: Any,
                      lora_model: Any,
                      eval_data: List[Dict],
                      sample_size: int = None) -> Dict:
        """Compare base model and LoRA adapter

        Args:
            base_model: Base model without LoRA
            lora_model: Model with LoRA adapter
            eval_data: Evaluation dataset
            sample_size: Number of samples to evaluate (None for all)

        Returns:
            Comparison results dictionary
        """
        if sample_size:
            eval_data = eval_data[:sample_size]

        logger.info(f"Comparing models on {len(eval_data)} examples")

        # Results storage
        base_results = []
        lora_results = []
        improvements = []  # Examples where LoRA is right but base is wrong
        regressions = []   # Examples where base is right but LoRA is wrong
        both_correct = []
        both_wrong = []

        # Evaluate each example
        for i, example in enumerate(tqdm(eval_data, desc="Comparing models")):
            # Evaluate base model
            base_result = self.evaluate_single(base_model, example)
            base_results.append(base_result)

            # Evaluate LoRA model
            lora_result = self.evaluate_single(lora_model, example)
            lora_results.append(lora_result)

            # Categorize
            if not base_result['is_correct'] and lora_result['is_correct']:
                improvements.append({
                    'index': i,
                    'question': example['question'],
                    'true_answer': example['answer'],
                    'base_answer': base_result['generated_answer'],
                    'lora_answer': lora_result['generated_answer']
                })
            elif base_result['is_correct'] and not lora_result['is_correct']:
                regressions.append({
                    'index': i,
                    'question': example['question'],
                    'true_answer': example['answer'],
                    'base_answer': base_result['generated_answer'],
                    'lora_answer': lora_result['generated_answer']
                })
            elif base_result['is_correct'] and lora_result['is_correct']:
                both_correct.append(i)
            else:
                both_wrong.append({
                    'index': i,
                    'question': example['question'],
                    'true_answer': example['answer'],
                    'base_answer': base_result['generated_answer'],
                    'lora_answer': lora_result['generated_answer']
                })

        # Calculate metrics
        base_accuracy = sum(r['is_correct'] for r in base_results) / len(base_results)
        lora_accuracy = sum(r['is_correct'] for r in lora_results) / len(lora_results)

        comparison = {
            'summary': {
                'total_examples': len(eval_data),
                'base_accuracy': base_accuracy,
                'lora_accuracy': lora_accuracy,
                'improvement': lora_accuracy - base_accuracy,
                'improvements_count': len(improvements),
                'regressions_count': len(regressions),
                'both_correct_count': len(both_correct),
                'both_wrong_count': len(both_wrong)
            },
            'improvements': improvements,
            'regressions': regressions,
            'both_wrong_sample': both_wrong[:10],  # Sample of failures
            'timestamp': datetime.now().isoformat()
        }

        return comparison

    def generate_report(self, comparison: Dict, variant_id: str = "") -> str:
        """Generate human-readable report

        Args:
            comparison: Comparison results from compare_models
            variant_id: ID of the LoRA variant

        Returns:
            Path to saved report
        """
        report_path = self.output_dir / f"comparison_report_{variant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Model Comparison Report\n\n")
            if variant_id:
                f.write(f"**LoRA Variant**: {variant_id}\n\n")
            f.write(f"**Date**: {comparison['timestamp']}\n\n")

            # Summary
            f.write("## Summary\n\n")
            summary = comparison['summary']
            f.write(f"- **Total Examples**: {summary['total_examples']}\n")
            f.write(f"- **Base Model Accuracy**: {summary['base_accuracy']:.2%}\n")
            f.write(f"- **LoRA Model Accuracy**: {summary['lora_accuracy']:.2%}\n")
            f.write(f"- **Improvement**: {summary['improvement']:.2%}\n\n")

            f.write("### Breakdown\n\n")
            f.write(f"- ✅ **Improvements** (Base wrong → LoRA correct): {summary['improvements_count']}\n")
            f.write(f"- ❌ **Regressions** (Base correct → LoRA wrong): {summary['regressions_count']}\n")
            f.write(f"- ✅✅ **Both Correct**: {summary['both_correct_count']}\n")
            f.write(f"- ❌❌ **Both Wrong**: {summary['both_wrong_count']}\n\n")

            # Improvements section
            f.write("## 🎯 Improvements (Base Wrong → LoRA Correct)\n\n")
            f.write("These examples show where the LoRA adapter fixed mistakes from the base model:\n\n")

            for i, imp in enumerate(comparison['improvements'][:20], 1):  # Show first 20
                f.write(f"### Example {i} (Index: {imp['index']})\n\n")
                f.write(f"**Question**: {imp['question']}\n\n")
                f.write(f"**Correct Answer**: {imp['true_answer']}\n\n")
                f.write(f"**Base Model Answer** ❌: {imp['base_answer']}\n\n")
                f.write(f"**LoRA Model Answer** ✅: {imp['lora_answer']}\n\n")
                f.write("---\n\n")

            if len(comparison['improvements']) > 20:
                f.write(f"*... and {len(comparison['improvements']) - 20} more improvements*\n\n")

            # Regressions section
            if comparison['regressions']:
                f.write("## ⚠️ Regressions (Base Correct → LoRA Wrong)\n\n")
                f.write("These examples show where the LoRA adapter introduced errors:\n\n")

                for i, reg in enumerate(comparison['regressions'][:10], 1):  # Show first 10
                    f.write(f"### Regression {i} (Index: {reg['index']})\n\n")
                    f.write(f"**Question**: {reg['question']}\n\n")
                    f.write(f"**Correct Answer**: {reg['true_answer']}\n\n")
                    f.write(f"**Base Model Answer** ✅: {reg['base_answer']}\n\n")
                    f.write(f"**LoRA Model Answer** ❌: {reg['lora_answer']}\n\n")
                    f.write("---\n\n")

            # Both wrong section (sample)
            if comparison['both_wrong_sample']:
                f.write("## 🔍 Sample of Examples Both Models Got Wrong\n\n")
                f.write("These examples remain challenging for both models:\n\n")

                for i, example in enumerate(comparison['both_wrong_sample'][:5], 1):
                    f.write(f"### Difficult Example {i} (Index: {example['index']})\n\n")
                    f.write(f"**Question**: {example['question']}\n\n")
                    f.write(f"**Correct Answer**: {example['true_answer']}\n\n")
                    f.write(f"**Base Model**: {example['base_answer']}\n\n")
                    f.write(f"**LoRA Model**: {example['lora_answer']}\n\n")
                    f.write("---\n\n")

            # Analysis section
            f.write("## 📊 Analysis\n\n")

            if summary['improvement'] > 0:
                f.write(f"The LoRA adapter improved accuracy by **{summary['improvement']:.2%}**, ")
                f.write(f"correctly answering **{summary['improvements_count']} questions** that the base model got wrong.\n\n")

                if summary['regressions_count'] > 0:
                    ratio = summary['improvements_count'] / summary['regressions_count']
                    f.write(f"For every regression introduced, the adapter fixed **{ratio:.1f} errors**, ")
                    f.write("showing a net positive improvement.\n\n")
                else:
                    f.write("**No regressions** were introduced - pure improvement!\n\n")
            else:
                f.write("The LoRA adapter did not improve overall accuracy.\n\n")

            # Recommendations
            f.write("## 💡 Recommendations\n\n")

            if summary['both_wrong_count'] > summary['total_examples'] * 0.3:
                f.write("- Many examples remain challenging. Consider:\n")
                f.write("  - Training for more epochs\n")
                f.write("  - Increasing LoRA rank\n")
                f.write("  - Using a larger training dataset\n\n")

            if summary['regressions_count'] > summary['improvements_count'] * 0.2:
                f.write("- Regression rate is concerning. Consider:\n")
                f.write("  - Reducing learning rate\n")
                f.write("  - Increasing regularization (dropout)\n")
                f.write("  - More careful hyperparameter tuning\n\n")

            if summary['improvement'] > 0.1:
                f.write("- **Excellent improvement!** This adapter is ready for deployment.\n")

        logger.info(f"Report saved to {report_path}")

        # Also save JSON version for programmatic access
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        return str(report_path)

    def generate_category_analysis(self,
                                  comparison: Dict,
                                  eval_data: List[Dict]) -> Dict:
        """Analyze improvements by question category

        Args:
            comparison: Comparison results
            eval_data: Original evaluation data with categories

        Returns:
            Category-wise analysis
        """
        category_stats = {}

        for imp in comparison['improvements']:
            idx = imp['index']
            if idx < len(eval_data) and 'category' in eval_data[idx]:
                category = eval_data[idx]['category']
                if category not in category_stats:
                    category_stats[category] = {'improvements': 0, 'total': 0}
                category_stats[category]['improvements'] += 1

        # Count totals per category
        for example in eval_data:
            if 'category' in example:
                category = example['category']
                if category not in category_stats:
                    category_stats[category] = {'improvements': 0, 'total': 0}
                category_stats[category]['total'] += 1

        # Calculate improvement rates
        for category in category_stats:
            total = category_stats[category]['total']
            if total > 0:
                category_stats[category]['improvement_rate'] = \
                    category_stats[category]['improvements'] / total

        return category_stats
