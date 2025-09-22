"""Solver Agent for R-Zero implementation - Learns reasoning through GRPO"""
from unsloth import FastModel
import os
import sys
import torch
import gc
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cli_formatter import CLIFormatter, SpinnerProgress

# Define reasoning format tokens (matching Teacher's format)
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


class SolverAgent:
    """Gemma-3-1B Solver that learns to reason step-by-step using GRPO

    Implements R-Zero paper's Solver with:
    - Self-consistency mechanism for answer extraction
    - RLVR (Reinforcement Learning with Verifiable Rewards)
    - Majority voting for robust pseudo-labels
    """

    def __init__(self, checkpoint_path=None):
        CLIFormatter.print_info("Initializing Solver (Gemma-3-1B with GRPO)...")

        self.max_seq_length = 2048  # Increased for reasoning steps

        # Load from checkpoint if provided, otherwise from base model
        if checkpoint_path and os.path.exists(checkpoint_path):
            CLIFormatter.print_info(f"Loading Solver from checkpoint: {checkpoint_path}")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
            )
        else:
            CLIFormatter.print_info("Loading base Gemma-3-1B model from Hugging Face")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name="unsloth/gemma-3-1b-it",
                max_seq_length=self.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
            )

        # Apply LoRA for GRPO (only if loading base model)
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,  # Important for reasoning
                finetune_mlp_modules=True,
                r=8,  # Following Unsloth notebook
                lora_alpha=8,
                lora_dropout=0,
                bias="none",
                random_state=3407,
            )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create system prompt for reasoning
        self.system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""

        # Compile regex patterns for extracting answers
        self.match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{REASONING_START}.+?{REASONING_END}.*?"
            rf"{SOLUTION_START}(.+?){SOLUTION_END}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )

        # Pattern for extracting numbers from solution
        self.match_numbers = re.compile(
            rf"{SOLUTION_START}.*?([\d\.]{{1,}})",
            flags=re.MULTILINE | re.DOTALL
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Solver loaded: {trainable:,} trainable params")

    def match_format_exactly(self, completions, **kwargs):
        """Reward function: Check if format matches exactly"""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly
            if self.match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(self, completions, **kwargs):
        """Reward function: Partial credit for format elements"""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Give partial credit for format elements, no negative penalties
            score += 0.5 if REASONING_START in response else 0
            score += 0.5 if REASONING_END in response else 0
            score += 0.5 if SOLUTION_START in response else 0
            score += 0.5 if SOLUTION_END in response else 0
            # Bonus for having them in correct order
            if REASONING_START in response and REASONING_END in response:
                if response.index(REASONING_START) < response.index(REASONING_END):
                    score += 0.5
            scores.append(score)
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):
        """Reward function: Check answer correctness"""
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            # Correct answer gets 3 points
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # Reward if answer is close via ratios
                try:
                    ratio = float(guess) / float(true_answer)
                    if ratio >= 0.9 and ratio <= 1.1:
                        score += 0.5
                    elif ratio >= 0.8 and ratio <= 1.2:
                        score += 0.25
                    else:
                        score -= 1.0  # Penalize wrong answers
                except:
                    score -= 0.5  # Penalize non-numeric
            scores.append(score)
        return scores

    def check_numbers(self, prompts, completions, answer, **kwargs):
        """Reward function: Extract and check numerical answers"""
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip())
                scores.append(1.5 if guess == true_answer else 0.0)
            except:
                scores.append(0)
                continue
        return scores

    def length_penalty(self, completions, **kwargs):
        """Reward function: Penalize overly long or short completions"""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            length = len(response.split())

            # Ideal length is 30-100 words
            if 30 <= length <= 100:
                score = 1.0
            elif length < 30:
                score = 0.5
            elif length <= 150:
                score = 0.3
            else:
                score = -0.5  # Way too long, penalize

            scores.append(score)
        return scores

    def generate_solution(self, question: str, temperature: float = 0.7) -> str:
        """Generate a single solution for a question

        Args:
            question: The problem to solve
            temperature: Sampling temperature

        Returns:
            Generated solution text
        """
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return answer

    def solve_with_self_consistency(
        self,
        question: str,
        m_samples: int = 10,
        temperature: float = 0.7
    ) -> Tuple[str, float, List[str]]:
        """Solve problem using self-consistency (majority vote)

        Args:
            question: Problem to solve
            m_samples: Number of solution samples
            temperature: Sampling temperature

        Returns:
            Tuple of (pseudo_label, empirical_accuracy, all_solutions)
        """
        solutions = []
        extracted_answers = []

        # Generate m solutions
        for _ in range(m_samples):
            solution = self.generate_solution(question, temperature)
            solutions.append(solution)

            # Extract numerical answer
            answer = self._extract_answer(solution)
            if answer:
                extracted_answers.append(answer)

        if not extracted_answers:
            return "", 0.0, solutions

        # Compute majority vote
        answer_counts = Counter(extracted_answers)
        most_common = answer_counts.most_common(1)[0]
        pseudo_label = most_common[0]
        empirical_accuracy = most_common[1] / len(extracted_answers)

        return pseudo_label, empirical_accuracy, solutions

    def _extract_answer(self, solution: str) -> str:
        """Extract numerical answer from solution

        Args:
            solution: Generated solution text

        Returns:
            Extracted numerical answer or empty string
        """
        # Try SOLUTION tags first
        if SOLUTION_START in solution and SOLUTION_END in solution:
            s_start = solution.find(SOLUTION_START) + len(SOLUTION_START)
            s_end = solution.find(SOLUTION_END)
            return solution[s_start:s_end].strip()

        # Fall back to last number
        numbers = re.findall(r'[-]?\d+(?:\.\d+)?', solution)
        return numbers[-1] if numbers else ""

    def solve_problems(self, problems: List[Dict], return_accuracy: bool = True) -> List[Dict]:
        """Generate solutions for problems with self-consistency

        Args:
            problems: List of problems to solve
            return_accuracy: Whether to compute overall accuracy

        Returns:
            List of results with solutions and correctness
        """
        results = []
        correct_count = 0

        CLIFormatter.print_info(f"Solving {len(problems)} problems...")

        with SpinnerProgress(f"Solving {len(problems)} problems") as spinner:
            for idx, problem in enumerate(problems):
                spinner.update(f"Solving problem {idx+1}/{len(problems)}")

                # Use self-consistency for robust answers
                pseudo_label, empirical_acc, solutions = self.solve_with_self_consistency(
                    problem['question'],
                    m_samples=5  # Reduced for speed during evaluation
                )

                # Check correctness if ground truth available
                is_correct = False
                if "answer" in problem and problem["answer"]:
                    try:
                        is_correct = abs(float(pseudo_label) - float(problem["answer"])) < 0.01
                        if is_correct:
                            correct_count += 1
                    except:
                        pass

                result = {
                    "question": problem["question"],
                    "solver_answer": pseudo_label,
                    "empirical_accuracy": empirical_acc,
                    "full_answers": solutions,
                    "ground_truth": problem.get("answer", ""),
                    "is_correct": is_correct,
                    "difficulty": problem.get("difficulty", 0.5)
                }
                results.append(result)

        # Calculate overall accuracy
        if return_accuracy and problems:
            accuracy = correct_count / len(problems)
            for r in results:
                r["accuracy"] = accuracy

        return results

    def train_with_grpo(self, problems: List[Dict], max_steps: int = 50):
        """Train Solver using GRPO to learn reasoning"""
        CLIFormatter.print_info("Training Solver with GRPO...")

        # Format dataset for GRPO
        dataset_items = []
        for problem in problems:
            dataset_items.append({
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": problem["question"]},
                ],
                "answer": problem["answer"],
            })

        dataset = Dataset.from_list(dataset_items)

        # GRPO configuration
        max_prompt_length = 128  # Reasonable prompt length
        max_completion_length = 150  # Limit completions to prevent rambling
        training_args = GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            logging_steps=1,
            per_device_train_batch_size=4,  # Higher batch size to prevent overfitting
            gradient_accumulation_steps=1,  # Less accumulation with higher batch
            num_generations=4,  # More diverse completions to explore solution space
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=max_steps,
            save_steps=max_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=f"outputs/grpo_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Configure generation for training
        generation_config = {
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Initialize GRPO trainer with reward functions
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.match_format_exactly,
                self.match_format_approximately,
                self.check_answer,
                self.check_numbers,
                self.length_penalty,  # Add length penalty
            ],
            args=training_args,
            train_dataset=dataset,
            generation_config=generation_config,
        )

        # Disable default verbose logging
        trainer.args.logging_steps = 1
        trainer.args.disable_tqdm = True  # Disable default progress bar

        # Track if we should suppress output
        original_log = trainer.log

        def formatted_log(logs, start_time=None):
            # Format and display key metrics
            if logs and isinstance(logs, dict):
                step = trainer.state.global_step

                # Check what type of log this is
                has_rewards = 'reward' in logs
                has_components = any(k.startswith('rewards/') for k in logs.keys())
                is_training_summary = 'train_runtime' in logs

                # Use beautiful formatting for training steps
                if step > 0 and (has_rewards or has_components):
                    # Use the new synthwave-styled GRPO formatter
                    CLIFormatter.format_grpo_stats(logs, step, max_steps)
                elif is_training_summary:
                    # Training summary
                    runtime = logs.get('train_runtime', 0)
                    CLIFormatter.print_success(f"GRPO training completed in {runtime:.1f}s")
                    if 'train_loss' in logs:
                        CLIFormatter.print_status("Final Loss", f"{logs['train_loss']:.6f}")
                    if 'epoch' in logs:
                        CLIFormatter.print_status("Epochs", f"{logs['epoch']:.2f}")
                    if 'train_steps_per_second' in logs:
                        CLIFormatter.print_status("Training Speed", f"{logs['train_steps_per_second']:.2f} steps/s")
                # Suppress raw dictionary output by not calling original_log
                # Only call for non-training logs
                elif not any(k in logs for k in ['loss', 'grad_norm', 'learning_rate']):
                    if start_time is not None:
                        original_log(logs, start_time)
                    else:
                        original_log(logs)

        trainer.log = formatted_log

        # Train with GRPO
        CLIFormatter.print_subheader(f"Phase 2: Solver GRPO Training")
        CLIFormatter.print_info(f"Training for {max_steps} steps to learn reasoning...")
        CLIFormatter.print_info("Watch the reward metrics to see learning progress!")
        print()  # Add spacing
        trainer.train()
        print()  # Add spacing after training

    def save_checkpoint(self, iteration: int, run_id: str = None, accuracy: float = None) -> str:
        """Save model checkpoint with metadata and return the path"""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"outputs/{run_id}/solver/iteration_{iteration}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        CLIFormatter.print_info(f"Saving Solver checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = {
            "iteration": iteration,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id
        }
        with open(f"{checkpoint_dir}/metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=2)

        return checkpoint_dir

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()
