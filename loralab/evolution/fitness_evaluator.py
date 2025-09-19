"""Fitness evaluator for LoRA variants

Evaluates the performance of LoRA variants on validation datasets.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Evaluates fitness of LoRA variants"""

    def __init__(self, model_manager):
        """Initialize evaluator

        Args:
            model_manager: ModelManager instance for tokenization
        """
        self.model_manager = model_manager
        self.tokenizer = model_manager.get_tokenizer()

    def evaluate(self,
                model: Any,
                eval_data: List[Dict],
                variant_id: str = "",
                max_samples: int = None,
                fast_mode: bool = True) -> Dict[str, float]:
        """Evaluate a model on the evaluation dataset

        Args:
            model: Model to evaluate
            eval_data: Evaluation dataset with 'question' and 'answer' fields
            variant_id: Identifier for logging
            max_samples: Maximum samples to evaluate (None for all)
            fast_mode: If True, only evaluate perplexity (skip accuracy)

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        device = next(model.parameters()).device

        # Limit samples if specified
        if max_samples:
            eval_data = eval_data[:max_samples]

        correct = 0
        total = len(eval_data)
        total_perplexity = 0

        # Increased batch size for faster evaluation
        batch_size = 32 if fast_mode else 16
        num_batches = (total + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {variant_id}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total)
                batch_data = eval_data[start_idx:end_idx]

                # Batch perplexity evaluation
                batch_perplexities = self._evaluate_perplexity_batch(model, batch_data, device)
                total_perplexity += sum(batch_perplexities)

                # Skip accuracy in fast mode (use perplexity only)
                if not fast_mode:
                    for item in batch_data:
                        accuracy_score = self._evaluate_accuracy(model, item, device)
                        correct += accuracy_score

        # Calculate metrics
        accuracy = correct / total if total > 0 and not fast_mode else 0.0
        avg_perplexity = total_perplexity / total if total > 0 else float('inf')

        metrics = {
            'accuracy': accuracy,
            'perplexity': avg_perplexity,
            'total_evaluated': total,
            'correct': correct,
            'fast_mode': fast_mode
        }

        if fast_mode:
            logger.info(f"Evaluation complete for {variant_id}: Perplexity={avg_perplexity:.2f} (fast mode)")
        else:
            logger.info(f"Evaluation complete for {variant_id}: "
                       f"Accuracy={accuracy:.2%}, Perplexity={avg_perplexity:.2f}")

        return metrics

    def _evaluate_accuracy(self, model, item: Dict, device) -> float:
        """Evaluate accuracy on a single item

        Args:
            model: Model to evaluate
            item: Data item with 'question' and 'answer'
            device: Device to run on

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        question = item.get('question', item.get('text', ''))
        true_answer = item.get('answer', item.get('label', '')).strip().lower()

        # Format prompt
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize with reduced max_length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256  # Reduced from 512
        ).to(device)

        # Generate answer with faster settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Reduced from 50
                temperature=0.1,
                do_sample=False,  # Greedy decoding is faster
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract answer
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated.replace(prompt, "").strip().lower()

        # Simple accuracy check - is the true answer contained in generated?
        # This is a simplified check - you might want more sophisticated matching
        if self._fuzzy_match(true_answer, generated_answer):
            return 1.0

        # Check for multiple choice style answers
        if len(true_answer) == 1 and true_answer in 'abcde':
            # Extract first letter that looks like an answer
            for char in generated_answer:
                if char in 'abcde':
                    return 1.0 if char == true_answer else 0.0

        return 0.0

    def _evaluate_perplexity_batch(self, model, batch_data: List[Dict], device) -> List[float]:
        """Evaluate perplexity on a batch of items

        Args:
            model: Model to evaluate
            batch_data: List of data items with 'question' and 'answer'
            device: Device to run on

        Returns:
            List of perplexity values
        """
        texts = []
        for item in batch_data:
            question = item.get('question', item.get('text', ''))
            answer = item.get('answer', item.get('label', ''))
            texts.append(f"Question: {question}\nAnswer: {answer}")

        # Batch tokenization
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256  # Reduced from 512
        ).to(device)

        # Calculate perplexity for batch
        with torch.no_grad():
            outputs = model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                labels=encodings['input_ids']
            )

            # Get per-sample loss using reduction='none'
            if hasattr(outputs, 'logits'):
                # Manual loss calculation for per-sample losses
                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = encodings['input_ids'][..., 1:].contiguous()
                loss_per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(shift_labels.size())

                # Average loss per sample
                loss_per_sample = loss_per_token.mean(dim=1)
                perplexities = torch.exp(loss_per_sample).tolist()
            else:
                # Fallback: use single loss value for all samples
                perplexity = torch.exp(outputs.loss).item()
                perplexities = [perplexity] * len(batch_data)

            # Cap perplexities
            perplexities = [min(p, 1000.0) for p in perplexities]

        return perplexities

    def _evaluate_perplexity(self, model, item: Dict, device) -> float:
        """Evaluate perplexity on a single item (fallback method)

        Args:
            model: Model to evaluate
            item: Data item with 'question' and 'answer'
            device: Device to run on

        Returns:
            Perplexity value
        """
        result = self._evaluate_perplexity_batch(model, [item], device)
        return result[0] if result else float('inf')

    def _fuzzy_match(self, true_answer: str, generated: str) -> bool:
        """Fuzzy matching for answers

        Args:
            true_answer: Ground truth answer
            generated: Generated answer

        Returns:
            True if answers match
        """
        # Direct containment check
        if true_answer in generated:
            return True

        # Check if all words from true answer are in generated
        true_words = set(true_answer.split())
        gen_words = set(generated.split())

        if len(true_words) > 0:
            overlap = len(true_words.intersection(gen_words))
            if overlap / len(true_words) >= 0.8:  # 80% word overlap
                return True

        # Numerical answer check
        try:
            true_num = float(true_answer)
            # Look for this number in generated text
            import re
            numbers = re.findall(r'-?\d+\.?\d*', generated)
            for num_str in numbers:
                if abs(float(num_str) - true_num) < 0.001:
                    return True
        except:
            pass

        return False

    def evaluate_batch(self,
                      models: List[Any],
                      eval_data: List[Dict],
                      variant_ids: List[str]) -> List[Dict[str, float]]:
        """Evaluate multiple models in sequence

        Args:
            models: List of models to evaluate
            eval_data: Evaluation dataset
            variant_ids: List of variant identifiers

        Returns:
            List of metric dictionaries
        """
        results = []

        for model, variant_id in zip(models, variant_ids):
            metrics = self.evaluate(model, eval_data, variant_id)
            results.append(metrics)

            # Clear cache between evaluations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def benchmark_speed(self, model: Any, num_samples: int = 10) -> Dict[str, float]:
        """Benchmark inference speed of a model

        Args:
            model: Model to benchmark
            num_samples: Number of samples to test

        Returns:
            Dictionary with speed metrics
        """
        model.eval()
        device = next(model.parameters()).device

        # Create dummy inputs
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "How do you make a peanut butter sandwich?",
        ] * (num_samples // 3 + 1)
        test_prompts = test_prompts[:num_samples]

        import time
        total_tokens = 0
        total_time = 0

        with torch.no_grad():
            for prompt in test_prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                start_time = time.time()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,  # Reduced for speed
                    temperature=0.7,
                    do_sample=False  # Greedy is faster
                )

                end_time = time.time()

                total_time += (end_time - start_time)
                total_tokens += outputs.shape[1] - inputs['input_ids'].shape[1]

        # Calculate metrics
        avg_time_per_sample = total_time / num_samples
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return {
            'avg_time_per_sample': avg_time_per_sample,
            'tokens_per_second': tokens_per_second,
            'total_time': total_time,
            'total_tokens': total_tokens
        }
