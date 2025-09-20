"""GRPO (Group Relative Policy Optimization) trainer for reasoning models

Implements GRPO training with reward functions for evolving LoRA adapters
that excel at reasoning and complex problem-solving tasks.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Callable, Tuple
import torch
from tqdm import tqdm
import numpy as np
from transformers import TrainerCallback
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

try:
    from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logging.warning("TRL not installed. Install with: pip install trl")

try:
    from ..utils.cli_formatter import CLIFormatter
    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Suppress default transformers logging to avoid raw dict output
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass


class FormattedLoggingCallback(TrainerCallback):
    """Custom callback to format training metrics using CLIFormatter"""

    def __init__(self, variant_id: str = "", use_formatter: bool = True):
        self.variant_id = variant_id
        self.step_count = 0
        self.total_steps = None
        self.use_formatter = use_formatter and FORMATTER_AVAILABLE
        self.last_log = {}

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.total_steps = state.max_steps if state else None
        if self.use_formatter and self.variant_id:
            CLIFormatter.print_subheader(f"Training {self.variant_id}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged"""
        if logs and self.use_formatter:
            # Filter out duplicate logs
            if logs != self.last_log:
                self.last_log = logs.copy()
                self.step_count = state.global_step if state else self.step_count + 1

                # Use CLIFormatter to display metrics
                if 'loss' in logs:
                    CLIFormatter.format_training_metrics(
                        logs,
                        step=self.step_count,
                        total_steps=self.total_steps
                    )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if self.use_formatter and self.variant_id:
            CLIFormatter.print_success(f"Training completed for {self.variant_id}")


class RewardFunctions:
    """Collection of reward functions for GRPO training"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize reward functions with configuration

        Args:
            config: Configuration for reward functions including:
                - reasoning_start: Token marking reasoning start
                - reasoning_end: Token marking reasoning end
                - solution_start: Token marking solution start
                - solution_end: Token marking solution end
        """
        self.config = config or {}

        # Default reasoning markers (can be customized)
        self.reasoning_start = self.config.get('reasoning_start', '<think>')
        self.reasoning_end = self.config.get('reasoning_end', '</think>')
        self.solution_start = self.config.get('solution_start', '<solution>')
        self.solution_end = self.config.get('solution_end', '</solution>')

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for matching format"""
        # Pattern to match the solution section
        self.solution_pattern = re.compile(
            rf"{re.escape(self.solution_start)}(.*?){re.escape(self.solution_end)}",
            flags=re.MULTILINE | re.DOTALL
        )

        # Pattern to match reasoning section
        self.reasoning_pattern = re.compile(
            rf"{re.escape(self.reasoning_start)}(.*?){re.escape(self.reasoning_end)}",
            flags=re.MULTILINE | re.DOTALL
        )

        # Pattern to extract numbers from solutions
        self.number_pattern = re.compile(
            r"[-+]?(?:\d*\.\d+|\d+\.?\d*)"
        )

    def format_reward(self, completions: List[List[Dict]], **kwargs) -> List[float]:
        """Reward for following the correct format

        Args:
            completions: List of completion responses

        Returns:
            List of reward scores
        """
        scores = []
        for completion in completions:
            response = completion[0]["content"] if completion else ""
            score = 0.0

            # Check for reasoning section
            if self.reasoning_pattern.search(response):
                score += 1.5

            # Check for solution section
            if self.solution_pattern.search(response):
                score += 1.5

            # Bonus for having both in correct order
            if self.reasoning_end in response and self.solution_start in response:
                reasoning_pos = response.find(self.reasoning_end)
                solution_pos = response.find(self.solution_start)
                if reasoning_pos < solution_pos:
                    score += 1.0

            scores.append(score)
        return scores

    def answer_accuracy_reward(self, prompts: List, completions: List,
                               answers: List[str], **kwargs) -> List[float]:
        """Reward based on answer accuracy

        Args:
            prompts: List of prompts
            completions: List of completions
            answers: List of correct answers

        Returns:
            List of reward scores
        """
        scores = []

        for completion, true_answer in zip(completions, answers):
            response = completion[0]["content"] if completion else ""

            # Extract answer from solution section
            solution_match = self.solution_pattern.search(response)
            if not solution_match:
                scores.append(-2.0)
                continue

            extracted_answer = solution_match.group(1).strip()

            # Try exact match first
            if extracted_answer == true_answer:
                scores.append(5.0)
            # Try normalized match (removing spaces)
            elif extracted_answer.replace(" ", "") == true_answer.replace(" ", ""):
                scores.append(4.0)
            # Try numeric comparison if both are numbers
            else:
                try:
                    extracted_num = float(self.number_pattern.search(extracted_answer).group())
                    true_num = float(self.number_pattern.search(true_answer).group())

                    # Calculate ratio
                    ratio = extracted_num / true_num if true_num != 0 else 0

                    if ratio == 1.0:
                        scores.append(3.5)
                    elif 0.95 <= ratio <= 1.05:
                        scores.append(2.0)
                    elif 0.9 <= ratio <= 1.1:
                        scores.append(1.0)
                    else:
                        scores.append(-1.0)
                except:
                    scores.append(-1.5)

        return scores

    def reasoning_quality_reward(self, completions: List[List[Dict]], **kwargs) -> List[float]:
        """Reward based on reasoning quality metrics

        Args:
            completions: List of completions

        Returns:
            List of reward scores
        """
        scores = []

        for completion in completions:
            response = completion[0]["content"] if completion else ""
            score = 0.0

            # Extract reasoning section
            reasoning_match = self.reasoning_pattern.search(response)
            if not reasoning_match:
                scores.append(-1.0)
                continue

            reasoning_text = reasoning_match.group(1)

            # Length reward (encourage detailed reasoning)
            words = reasoning_text.split()
            if len(words) > 50:
                score += 1.0
            elif len(words) > 20:
                score += 0.5

            # Step indicators (looking for structured thinking)
            step_indicators = ['first', 'second', 'then', 'next', 'finally',
                              'step 1', 'step 2', '1.', '2.', '3.']
            step_count = sum(1 for indicator in step_indicators
                            if indicator.lower() in reasoning_text.lower())
            score += min(step_count * 0.3, 1.5)

            # Mathematical operations mentioned
            math_terms = ['add', 'subtract', 'multiply', 'divide', 'calculate',
                         'sum', 'total', 'equals', '=', '+', '-', '*', '/']
            math_count = sum(1 for term in math_terms
                            if term in reasoning_text.lower())
            score += min(math_count * 0.2, 1.0)

            scores.append(score)

        return scores

    def combined_reward(self, prompts: List, completions: List,
                       answers: List[str], weights: Dict[str, float] = None,
                       **kwargs) -> List[float]:
        """Combine multiple reward functions with weights

        Args:
            prompts: List of prompts
            completions: List of completions
            answers: List of correct answers
            weights: Dictionary of weights for each reward function

        Returns:
            List of combined reward scores
        """
        if weights is None:
            weights = {
                'format': 1.0,
                'accuracy': 2.0,
                'reasoning': 1.5
            }

        # Calculate individual rewards
        format_rewards = self.format_reward(completions)
        accuracy_rewards = self.answer_accuracy_reward(prompts, completions, answers)
        reasoning_rewards = self.reasoning_quality_reward(completions)

        # Combine with weights
        combined = []
        for f, a, r in zip(format_rewards, accuracy_rewards, reasoning_rewards):
            score = (weights['format'] * f +
                    weights['accuracy'] * a +
                    weights['reasoning'] * r)
            combined.append(score)

        return combined


class GRPOTrainer:
    """GRPO trainer for reasoning-focused LoRA evolution"""

    def __init__(self, model_manager, training_config: Dict[str, Any], full_config: Dict[str, Any] = None):
        """Initialize GRPO trainer

        Args:
            model_manager: UnslothModelManager or ModelManager instance
            training_config: Training configuration
            full_config: Full configuration (optional, for accessing evolution config)
        """
        if not TRL_AVAILABLE:
            raise ImportError("TRL is required for GRPO training. Install with: pip install trl")

        self.model_manager = model_manager
        self.training_config = training_config
        self.full_config = full_config or {}
        self.tokenizer = model_manager.get_tokenizer()
        self.reward_functions = RewardFunctions(training_config.get('grpo', {}))

        # Use evolution.generations from full config if available
        evolution_config = self.full_config.get('evolution', {})
        self.num_generations = evolution_config.get('generations',
                                training_config.get('grpo', {}).get('num_generations', 4))

    def prepare_dataset_for_grpo(self, dataset: List[Dict]) -> List[Dict]:
        """Prepare dataset for GRPO training

        Args:
            dataset: Raw dataset with questions and answers

        Returns:
            Formatted dataset for GRPO
        """
        grpo_dataset = []

        for item in dataset:
            # Extract question and answer
            question = item.get('question', item.get('text', ''))
            answer = item.get('answer', item.get('label', ''))

            # Format as GRPO expects
            formatted = {
                'prompt': [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": question}
                ],
                'answer': answer
            }
            grpo_dataset.append(formatted)

        return grpo_dataset

    def _get_system_prompt(self) -> str:
        """Get system prompt for reasoning tasks"""
        reasoning_start = self.reward_functions.reasoning_start
        reasoning_end = self.reward_functions.reasoning_end
        solution_start = self.reward_functions.solution_start
        solution_end = self.reward_functions.solution_end

        return f"""You are a reasoning assistant. Think through problems step-by-step.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then provide your final answer between {solution_start} and {solution_end}."""

    def train(self, model: Any, train_data: List[Dict],
             learning_rate: float, max_steps: int = 100,
             variant_id: str = "") -> Dict[str, float]:
        """Train model using GRPO

        Args:
            model: Model with LoRA applied
            train_data: Training dataset
            learning_rate: Learning rate
            max_steps: Maximum training steps
            variant_id: Identifier for logging

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting GRPO training for {variant_id}")

        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        # Prepare dataset
        grpo_dataset = self.prepare_dataset_for_grpo(train_data)

        # GRPO configuration
        grpo_config = self.training_config.get('grpo', {})
        max_prompt_length = grpo_config.get('max_prompt_length', 512)
        max_completion_length = grpo_config.get('max_completion_length', 512)

        # Create GRPO training arguments
        training_args = GRPOConfig(
            temperature=grpo_config.get('temperature', 1.0),
            learning_rate=learning_rate,
            weight_decay=self.training_config.get('weight_decay', 0.01),
            warmup_ratio=self.training_config.get('warmup_ratio', 0.1),
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            logging_strategy="steps",
            per_device_train_batch_size=self.training_config.get('batch_size', 1),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
            num_generations=self.num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=max_steps,
            save_steps=max_steps,  # Save only at the end
            report_to="none",
            output_dir=f"outputs/grpo_{variant_id}",
            disable_tqdm=True,  # Disable progress bars to reduce clutter
        )

        # Get reward function weights
        reward_weights = grpo_config.get('reward_weights', {
            'format': 1.0,
            'accuracy': 2.0,
            'reasoning': 1.5
        })

        # Create reward functions list
        # GRPO passes completions as first arg, and dataset columns as kwargs
        # The 'answer' field from the dataset will be passed in kwargs
        def custom_reward_func(completions, answer=None, **kwargs):
            """Custom reward function that uses our RewardFunctions class"""
            import re

            # Completions come as list of message dicts, extract just the content
            completion_texts = []
            for comp in completions:
                if isinstance(comp, list) and len(comp) > 0:
                    # Extract content from message dict
                    completion_texts.append(comp[0].get('content', ''))
                else:
                    completion_texts.append(str(comp))

            # Convert answer to list if it's not already
            if answer is not None and not isinstance(answer, list):
                answer = [answer] * len(completions)

            rewards = []
            for i, text in enumerate(completion_texts):
                reward = 0.0

                # Basic reward for generating any response
                if len(text.strip()) > 10:
                    reward += 0.1

                # Reward for attempting math (numbers present)
                if re.search(r'\d+', text):
                    reward += 0.2

                # Reward for showing work (multiple lines/steps)
                lines = text.strip().split('\n')
                if len(lines) > 2:
                    reward += 0.2

                # Reward for mathematical operations
                if any(op in text for op in ['+', '-', '*', '/', '=']):
                    reward += 0.2

                # Reward for keywords showing reasoning
                reasoning_keywords = ['because', 'therefore', 'so', 'first', 'then',
                                     'step', 'calculate', 'total', 'answer', 'result']
                keyword_count = sum(1 for keyword in reasoning_keywords if keyword.lower() in text.lower())
                reward += min(keyword_count * 0.1, 0.3)

                # Try to extract final answer and check accuracy if possible
                if answer and i < len(answer):
                    correct_answer = str(answer[i])
                    # Look for the answer number in the completion
                    if correct_answer in text:
                        reward += 1.0  # Big reward for correct answer

                # Cap reward between -1 and 2
                reward = max(-1.0, min(2.0, reward))
                rewards.append(reward)

            # Log first completion for debugging (only occasionally)
            import random
            if random.random() < 0.1 and completion_texts:  # Log 10% of the time
                logger.debug(f"Sample completion: {completion_texts[0][:200]}...")
                logger.debug(f"Reward: {rewards[0]}")

            return rewards

        reward_funcs = [custom_reward_func]

        # Create custom callback for formatted logging
        formatted_callback = FormattedLoggingCallback(variant_id=variant_id)

        # Create GRPO trainer with custom callback
        trainer = TRLGRPOTrainer(
            model=model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=grpo_dataset,
            callbacks=[formatted_callback] if FORMATTER_AVAILABLE else [],
        )

        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        # Train and collect metrics
        train_output = trainer.train()

        # Extract metrics - look for various reward keys
        # Debug: show all available metrics keys
        logger.debug(f"All metric keys: {list(train_output.metrics.keys())}")
        reward_keys = [k for k in train_output.metrics.keys() if 'reward' in k.lower()]
        if reward_keys:
            logger.debug(f"Available reward keys: {reward_keys}")
            # Try to get the first reward key found
            for key in reward_keys:
                reward_value = train_output.metrics.get(key, 0.0)
                if reward_value != 0.0:
                    logger.debug(f"Using reward from key '{key}': {reward_value}")
                    break
        else:
            # Fallback to known keys
            reward_value = train_output.metrics.get('train_reward',
                          train_output.metrics.get('train_rewards/custom_reward_func/mean',
                          train_output.metrics.get('reward',
                          train_output.metrics.get('rewards/custom_reward_func/mean', 0.0))))

        metrics = {
            'final_loss': train_output.metrics.get('train_loss', float('inf')),
            'total_steps': train_output.metrics.get('train_steps', 0),
            'rewards': reward_value,
        }

        logger.info(f"GRPO training completed for {variant_id}")
        logger.info(f"  Final loss: {metrics['final_loss']:.4f}")
        logger.info(f"  Average reward: {metrics['rewards']:.4f}")

        return metrics

    def pre_train_formatting(self, model: Any, format_examples: List[Dict],
                           learning_rate: float = 2e-4, epochs: int = 2) -> float:
        """Pre-train model on formatting before GRPO

        This helps GRPO focus on reasoning rather than format learning.

        Args:
            model: Model to pre-train
            format_examples: Examples with correct formatting
            learning_rate: Learning rate for pre-training
            epochs: Number of pre-training epochs

        Returns:
            Final pre-training loss
        """
        logger.info("Pre-training model on format examples...")

        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            logger.warning("SFTTrainer not available, skipping pre-training")
            return 0.0

        # Prepare formatted examples
        formatted_texts = []
        for example in format_examples:
            question = example.get('question', '')
            answer = example.get('answer', '')
            reasoning = example.get('reasoning', 'Let me think about this step by step.')

            text = f"""{self._get_system_prompt()}

  User: {question}
  Assistant: {self.reward_functions.reasoning_start}{reasoning}{self.reward_functions.reasoning_end}
  {self.reward_functions.solution_start}{answer}{self.reward_functions.solution_end}"""
            formatted_texts.append(text)

        # Create SFT training config
        # For pre-training with small datasets, use batch size of 1 to get more training steps
        num_examples = len(formatted_texts)

        # Use batch size of 1 for pre-training to ensure proper training
        batch_size = 1

        # Keep gradient accumulation small for pre-training
        grad_accum_steps = min(2, num_examples)  # Max 2 for pre-training

        sft_config = SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=1,  # Force single process for Windows compatibility
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=1,
            logging_strategy="steps",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            disable_tqdm=True,  # Disable progress bars for cleaner output
        )

        # Calculate training steps
        effective_batch_size = batch_size * grad_accum_steps
        # For pre-training, each example should be seen individually
        steps_per_epoch = len(formatted_texts)  # One step per example with batch_size=1
        total_steps = steps_per_epoch * epochs

        logger.info(f"Pre-training configuration:")
        logger.info(f"  - Examples: {len(formatted_texts)}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Gradient accumulation: {grad_accum_steps}")
        logger.info(f"  - Effective batch size: {effective_batch_size}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Expected training steps: {total_steps}")

        # Create dataset
        from datasets import Dataset
        pre_train_dataset = Dataset.from_dict({"text": formatted_texts})

        logger.info(f"Created pre-training dataset with {len(pre_train_dataset)} examples")

        # Create custom callback for formatted logging
        formatted_callback = FormattedLoggingCallback(variant_id="pre-training")

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=pre_train_dataset,
            args=sft_config,
            callbacks=[formatted_callback] if FORMATTER_AVAILABLE else [],
        )
        
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        # Train
        train_output = trainer.train()

        final_loss = train_output.metrics.get('train_loss', 0.0)
        logger.info(f"Pre-training completed with final loss: {final_loss:.4f}")

        return final_loss
