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

        # GRPO num_generations is different from evolution generations
        # It's the number of completions to generate per prompt for comparison
        grpo_config = training_config.get('grpo', {})
        self.num_generations = grpo_config.get('num_generations', 2)  # Minimum 2 required for GRPO

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

        return f"""You are a mathematical reasoning assistant. Solve problems step-by-step.

Format your response as follows:
1. Show your step-by-step reasoning and calculations
2. End with a clear final answer

IMPORTANT: Your last line must be: "Final Answer: [number]"

For example:
- If the answer is 42, end with: "Final Answer: 42"
- If the answer is $1,250, end with: "Final Answer: 1250"
- If the answer is 3.5, end with: "Final Answer: 3.5"

Always provide just the numeric value in the final answer, without units or symbols."""

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

        # Disable torch compile for GRPO to avoid dynamo issues
        #os.environ['TORCHDYNAMO_DISABLE'] = '1'
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        # Prepare dataset
        grpo_dataset = self.prepare_dataset_for_grpo(train_data)

        # GRPO configuration
        grpo_config = self.training_config.get('grpo', {})
        max_prompt_length = grpo_config.get('max_prompt_length', 512)
        max_completion_length = grpo_config.get('max_completion_length', 512)

        # For GRPO, batch size must be a multiple of num_generations
        # Use 1 for stability with small models
        batch_size = 1

        # Create GRPO training arguments with valid parameters only
        training_args = GRPOConfig(
            # Generation parameters
            temperature=grpo_config.get('temperature', 0.5),  # Lower for stability
            top_k=grpo_config.get('top_k', 30),  # More restricted sampling
            top_p=grpo_config.get('top_p', 0.85),  # Tighter nucleus sampling
            # Training parameters - VERY conservative for stability
            learning_rate=learning_rate * 0.1,  # Much lower LR for small model stability
            weight_decay=self.training_config.get('weight_decay', 0.1),  # Strong regularization
            warmup_ratio=self.training_config.get('warmup_ratio', 0.3),  # More warmup
            lr_scheduler_type="constant_with_warmup",  # Safer schedule
            optim="adamw_8bit",
            logging_steps=1,
            logging_strategy="steps",
            per_device_train_batch_size=batch_size,  # Must be 1 or multiple of num_generations
            gradient_accumulation_steps=2,  # Reduced for stability
            num_generations=self.num_generations,  # Number of completions per prompt
            max_prompt_length=min(max_prompt_length, 384),  # Allow longer for word problems
            max_completion_length=min(max_completion_length, 256),  # Longer for solutions
            max_steps=max_steps,
            save_steps=max_steps,  # Save only at the end
            # Stability settings - CRITICAL
            max_grad_norm=0.3,  # Very aggressive gradient clipping
            beta=grpo_config.get('beta', 0.5),  # High KL penalty
            epsilon=grpo_config.get('epsilon', 0.1),  # Small clipping epsilon
            # Don't use loss_type or scale_rewards - they cause errors in Unsloth
            mask_truncated_completions=True,  # Exclude truncated completions from loss
            # Misc settings
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
            """Custom reward function with better discrimination"""
            import re
            import math

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

                # Penalty for too short or too long responses
                text_len = len(text.strip())
                if text_len < 20:
                    reward -= 3.0  # Strong penalty for trivial responses
                elif text_len > 1500:
                    reward -= 1.0  # Penalty for rambling
                else:
                    # Good length bonus
                    reward += 0.5

                # Check for step-by-step reasoning (reduced weight)
                lines = text.strip().split('\n')
                if len(lines) >= 3:
                    reward += 0.5  # Smaller reward for multi-line reasoning
                elif len(lines) == 1:
                    reward -= 1.0  # Penalize single-line answers for math problems

                # Look for explicit step indicators
                step_patterns = [r'step\s*\d+', r'first\s*,', r'then\s*,', r'finally\s*,',
                               r'^\d+\.', r'^\d+\)']
                step_count = 0
                for pattern in step_patterns:
                    if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                        step_count += 1
                reward += min(step_count * 0.3, 1.0)  # Reduced: Up to 1.0 for structured steps

                # Mathematical operations (reduced weight)
                math_ops_found = 0
                for op in ['+', '-', '*', '/', '=']:
                    if op in text:
                        math_ops_found += text.count(op)
                if math_ops_found > 0:
                    reward += min(math_ops_found * 0.2, 1.0)  # Reduced: Up to 1.0 for math operations
                else:
                    reward -= 2.0  # Penalty for no math operations in a math problem

                # Check for numbers in the response (should have calculations)
                numbers = re.findall(r'\d+\.?\d*', text)
                if len(numbers) >= 3:  # Multiple numbers suggest working
                    reward += 0.5  # Reduced reward
                elif len(numbers) <= 1:
                    reward -= 2.0  # Too few numbers for a math problem

                # CRITICAL: Check answer accuracy (biggest weight)
                if answer and i < len(answer):
                    correct_answer_full = str(answer[i]).strip()

                    # Extract the final numeric answer from GSM8K format
                    if "####" in correct_answer_full:
                        correct_answer = correct_answer_full.split("####")[-1].strip()
                    else:
                        correct_answer = correct_answer_full

                    # Try to find the answer in various formats
                    answer_found = False

                    # IMPORTANT: For GSM8K, we need to check for the FINAL answer, not intermediate values
                    # Look for answer patterns that indicate a final answer
                    # Prioritize our preferred format
                    final_answer_pattern = rf'Final Answer:\s*{re.escape(correct_answer)}(?:\D|$)'

                    # Check for our preferred format first
                    if re.search(final_answer_pattern, text, re.IGNORECASE):
                        answer_found = True
                        reward += 12.0  # Highest reward for following exact format
                    else:
                        # Fallback patterns for other formats
                        final_answer_patterns = [
                            rf'answer[:\s]*\$?{re.escape(correct_answer)}(?:\D|$)',  # "answer: 840" or "answer: $840"
                            rf'total[:\s]*\$?{re.escape(correct_answer)}(?:\D|$)',   # "total: 840"
                            rf'={re.escape(correct_answer)}(?:\D|$)',         # "= 840" (not part of longer number)
                            rf'\$?{re.escape(correct_answer)}(?:\D|$).*$',   # Answer at very end
                        ]

                        # Check if final answer appears in proper context
                        for pattern in final_answer_patterns:
                            if re.search(pattern, text, re.IGNORECASE):
                                answer_found = True
                                # Lower rewards for non-standard formats
                                if re.search(rf'answer[:\s]*\$?{re.escape(correct_answer)}', text, re.IGNORECASE):
                                    reward += 9.0  # Good - has "answer" keyword
                                elif text.strip().endswith(correct_answer):
                                    reward += 8.0  # OK - answer at the end
                                else:
                                    reward += 6.0  # Acceptable - answer present
                                break

                    # If not found, try numeric matching with stricter context
                    if not answer_found:
                        try:
                            # Try to parse the correct answer as a number
                            correct_num = float(correct_answer.replace(',', ''))

                            # Look for numbers in answer context only
                            # Focus on the last part of the text where answers typically appear
                            text_lower = text.lower()
                            answer_section = text

                            # Try to isolate the answer section
                            for keyword in ['answer', 'therefore', 'total', 'result', 'so']:
                                if keyword in text_lower:
                                    # Get everything after the last occurrence of the keyword
                                    parts = text_lower.split(keyword)
                                    if len(parts) > 1:
                                        answer_section = text[text_lower.rfind(keyword):]
                                        break

                            # Extract numbers from the answer section
                            gen_numbers = re.findall(r'-?\d+\.?\d*', answer_section)

                            # Check if any generated number matches
                            for idx, gen_num_str in enumerate(gen_numbers):
                                try:
                                    gen_num = float(gen_num_str)
                                    if abs(gen_num - correct_num) < 0.1:  # Small tolerance
                                        answer_found = True
                                        # Higher reward if it's the last number in answer section
                                        if idx == len(gen_numbers) - 1:
                                            reward += 8.0  # Last number in answer section
                                        else:
                                            reward += 4.0  # Number appears in answer section
                                        break
                                except:
                                    continue

                        except (ValueError, AttributeError):
                            pass

                    # Heavy penalty for wrong or missing answer
                    if not answer_found:
                        # Strong base penalty for wrong answer
                        reward -= 10.0  # Base penalty for being wrong

                        # Additional penalties based on context
                        if text.endswith("...") or len(text) > 250:
                            # Slightly less penalty if truncated - might have been on track
                            reward += 2.0  # Reduce penalty by 2 (net -8)
                        elif re.search(r'answer|total|result|therefore|=\s*\d+', text.lower()):
                            # They tried to give an answer but it was wrong
                            reward += 1.0  # Reduce penalty by 1 (net -9)
                        # else: full -10 penalty for no answer attempt

                # Format bonus - encourage "Final Answer:" format
                if re.search(r'Final Answer:\s*\d', text, re.IGNORECASE):
                    reward += 2.0  # Bonus for using the correct format structure

                # Reasoning quality keywords (smaller weight)
                reasoning_indicators = ['because', 'therefore', 'thus', 'since', 'so we',
                                      'this means', 'calculate', 'multiply', 'divide', 'add', 'subtract']
                reasoning_count = sum(1 for word in reasoning_indicators
                                    if word.lower() in text.lower())
                reward += min(reasoning_count * 0.2, 1.0)

                # Use RewardFunctions format checking (smaller weight)
                if self.reward_functions.reasoning_pattern.search(text):
                    reward += 1.0  # Bonus for using reasoning tags
                if self.reward_functions.solution_pattern.search(text):
                    reward += 1.0  # Bonus for using solution tags

                # Final reward range: approximately -15 to +15
                # This provides much better discrimination
                rewards.append(reward)

            # Enhanced logging for debugging with CLI formatting
            import random
            if random.random() < 0.3:  # Log 30% of the time for better debugging
                try:
                    from ..utils.cli_formatter import CLIFormatter, Fore, Style
                    use_formatter = True
                except ImportError:
                    use_formatter = False

                # Extract model's answer from the completion
                def extract_model_answer(text):
                    """Extract the model's final answer from the completion"""
                    # Look for "Final Answer:" pattern first
                    final_match = re.search(r'Final Answer:\s*([-\d.]+)', text, re.IGNORECASE)
                    if final_match:
                        return final_match.group(1)

                    # Look for answer patterns
                    answer_patterns = [
                        r'answer[:\s]+\$?([-\d.]+)',
                        r'total[:\s]+\$?([-\d.]+)',
                        r'result[:\s]+\$?([-\d.]+)',
                        r'=\s*([-\d.]+)\s*$',
                    ]

                    for pattern in answer_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            return match.group(1)

                    # Last resort: find last number in text
                    numbers = re.findall(r'[-\d.]+', text)
                    if numbers:
                        return numbers[-1]
                    return "NOT FOUND"

                # Clean text for ASCII-safe logging
                sample_text = completion_texts[0] if completion_texts else 'None'
                # Replace problematic Unicode characters
                sample_text = sample_text.replace('\u2212', '-').replace('−', '-').replace('–', '-')
                sample_text = sample_text.encode('ascii', 'replace').decode('ascii')

                # Extract answers
                model_answer = extract_model_answer(sample_text)
                target_answer = None
                if answer and len(answer) > 0:
                    full_answer = str(answer[0])
                    if "####" in full_answer:
                        target_answer = full_answer.split("####")[-1].strip()
                    else:
                        target_answer = full_answer

                # Check if correct
                is_correct = False
                if target_answer and model_answer != "NOT FOUND":
                    try:
                        is_correct = abs(float(model_answer) - float(target_answer)) < 0.1
                    except:
                        is_correct = model_answer == target_answer

                if use_formatter:
                    # Colored output using CLI formatter
                    print()  # New line for spacing
                    CLIFormatter.print_box_start("GRPO Reward Evaluation", color=Fore.CYAN)

                    # Show response preview
                    print(f"{Fore.WHITE}Response Preview (last 300 chars):{Style.RESET_ALL}")
                    print(f"{Fore.LIGHTBLACK_EX}{sample_text[-300:]}...{Style.RESET_ALL}")
                    print()

                    # Show extracted vs target answer
                    answer_color = Fore.GREEN if is_correct else Fore.RED
                    print(f"{Fore.CYAN}Model's Answer: {answer_color}{model_answer}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Target Answer:  {Fore.YELLOW}{target_answer if target_answer else 'N/A'}{Style.RESET_ALL}")

                    # Show correctness
                    if is_correct:
                        print(f"{Fore.GREEN}CORRECT{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}INCORRECT{Style.RESET_ALL}")

                    # Show reward with clarity about what it represents
                    reward_color = Fore.GREEN if rewards[0] > 5 else (Fore.YELLOW if rewards[0] > 0 else Fore.RED)
                    print()
                    print(f"{Fore.CYAN}This Sample's Reward: {reward_color}{rewards[0]:.2f}{Style.RESET_ALL} / 15.00")

                    # Show average if we have multiple rewards
                    if len(rewards) > 1:
                        avg_reward = sum(rewards) / len(rewards)
                        avg_color = Fore.GREEN if avg_reward > 5 else (Fore.YELLOW if avg_reward > 0 else Fore.RED)
                        print(f"{Fore.CYAN}Batch Average Reward: {avg_color}{avg_reward:.2f}{Style.RESET_ALL}")

                    CLIFormatter.print_box_end()
                else:
                    # Fallback to logger
                    logger.info(f"[GRPO Reward] Sample completion (first 400 chars):")
                    logger.info(f"  Text: {sample_text[:400]}...")
                    logger.info(f"  Model's Answer: {model_answer}")
                    logger.info(f"  Target Answer: {target_answer if target_answer else 'N/A'}")
                    logger.info(f"  Correct: {'YES' if is_correct else 'NO'}")
                    logger.info(f"  Reward: {rewards[0]:.2f} (range: -15 to +15)")

            return rewards

        reward_funcs = [custom_reward_func]

        # Create custom callback for formatted logging
        formatted_callback = FormattedLoggingCallback(variant_id=variant_id)

        # Create GRPO trainer with custom callback
        try:
            trainer = TRLGRPOTrainer(
                model=model,
                processing_class=self.tokenizer,
                reward_funcs=reward_funcs,
                args=training_args,
                train_dataset=grpo_dataset,
                callbacks=[formatted_callback] if FORMATTER_AVAILABLE else [],
            )
        except Exception as e:
            logger.error(f"Failed to initialize GRPO trainer: {e}")
            logger.error("GRPO training appears to be unstable on this system. Consider using SFT mode instead.")
            # Return failure metrics
            return {
                'final_loss': float('inf'),
                'total_steps': 0,
                'rewards': 0.0,
                'error': f"GRPO initialization failed: {str(e)}"
            }

        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        # Train and collect metrics with error handling
        try:
            train_output = trainer.train()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.error(f"CUDA error during GRPO training: {e}")
            logger.error("This may be due to GRPO incompatibility. Try: 1) Using SFT mode, 2) Reducing batch size, 3) Disabling torch.compile")
            # Try to recover
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Return failure metrics
            return {
                'final_loss': float('inf'),
                'total_steps': 0,
                'rewards': 0.0,
                'error': str(e)
            }

        # Extract metrics - look for various reward keys
        # The reward metrics are in the state's log history, not the final metrics
        reward_value = 0.0
        reward_found = False

        # Check trainer state for logged metrics
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
            # Get the last non-empty log entry that has reward data
            for log_entry in reversed(trainer.state.log_history):
                if isinstance(log_entry, dict):
                    # Try different reward keys
                    reward_keys = [
                        'reward',  # Simple reward key
                        'rewards/custom_reward_func/mean',  # Custom reward mean
                        'train_reward',  # Main training reward
                        'train_rewards/custom_reward_func/mean',  # With train prefix
                    ]

                    for key in reward_keys:
                        if key in log_entry:
                            reward_value = log_entry[key]
                            logger.info(f"Found reward value {reward_value:.4f} from key '{key}' in log history")
                            reward_found = True
                            break

                    if reward_found:
                        break

        # Fallback to checking train_output.metrics
        if not reward_found:
            all_keys = list(train_output.metrics.keys())
            logger.debug(f"Available metrics in train_output: {[k for k in all_keys if 'reward' in k.lower()]}")

            for key in ['train_reward', 'reward', 'rewards_mean']:
                if key in train_output.metrics:
                    reward_value = train_output.metrics[key]
                    logger.info(f"Found reward value {reward_value:.4f} from key '{key}' in train_output.metrics")
                    reward_found = True
                    break

        if not reward_found:
            logger.warning("No reward metric found in training output, defaulting to 0.0")

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
