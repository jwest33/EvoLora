"""Challenger Agent for R-Zero implementation - Trainable model that generates problems using GRPO"""
from unsloth import FastModel
import os
import sys
import torch
import gc
import re
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple
from collections import Counter
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import json

# Globally suppress padding warnings that persist despite correct configuration
warnings.filterwarnings("ignore", message=".*decoder-only.*right-padding.*")
warnings.filterwarnings("ignore", message=".*right-padding.*")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cli_formatter import CLIFormatter, SpinnerProgress


def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


class ChallengerAgent:
    """Trainable Challenger that learns to generate problems at optimal difficulty using GRPO"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize Challenger from base model or checkpoint

        Args:
            checkpoint_path: Path to existing checkpoint, or None to load base model
        """
        CLIFormatter.print_info("Initializing Challenger (Gemma-3-1B with GRPO)...")

        self.max_seq_length = 2048

        # Load from checkpoint if provided, otherwise from base model
        if checkpoint_path and os.path.exists(checkpoint_path):
            CLIFormatter.print_info(f"Loading Challenger from checkpoint: {checkpoint_path}")
            # Suppress padding warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*right-padding.*")
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name=checkpoint_path,
                    max_seq_length=self.max_seq_length,
                    load_in_4bit=False,
                    load_in_8bit=False,
                )
        else:
            CLIFormatter.print_info("Loading base Gemma-3-1B model for Challenger")
            # Suppress padding warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*right-padding.*")
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name="unsloth/gemma-3-1b-it",
                    max_seq_length=self.max_seq_length,
                    load_in_4bit=False,
                    load_in_8bit=False,
                    full_finetuning=False,
                )

            # Apply LoRA for GRPO
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=8,  # Following paper's configuration
                lora_alpha=8,
                lora_dropout=0,
                bias="none",
                random_state=3407,
            )

        # Set pad token and padding side for decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: Always set left padding for decoder-only models
        # This needs to be set every time, even when loading from checkpoint
        self.tokenizer.padding_side = "left"

        # System prompt for problem generation
        self.system_prompt = """You are an expert competition-math problem setter. Generate a challenging but solvable math problem.
The problem should:
1. Be a word problem with real-world context
2. Require mathematical reasoning
3. Have a clear numerical answer

Format your output exactly as:
<question>
[Problem statement here]
</question>
Answer: [numerical answer only]"""

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Challenger loaded: {trainable:,} trainable params")

        # Initialize problem history for deduplication
        self.problem_history = set()
        self.iteration = 0

    def generate_candidate_problems(
        self,
        num_problems: int,
        temperature: float = 0.8,
        batch_size: int = 4
    ) -> List[Dict]:
        """Generate candidate problems using current policy

        Args:
            num_problems: Number of problems to generate
            temperature: Sampling temperature
            batch_size: Batch size for generation

        Returns:
            List of generated problems with questions and extracted answers
        """
        problems = []

        with SpinnerProgress(f"Generating {num_problems} candidate problems") as spinner:
            for batch_idx in range(0, num_problems, batch_size):
                current_batch_size = min(batch_size, num_problems - batch_idx)
                spinner.update(f"Generating problems {batch_idx+1}-{batch_idx+current_batch_size}/{num_problems}")

                # Generate batch of problems
                prompts = []
                for _ in range(current_batch_size):
                    prompt = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": "Generate a new math problem:"}
                    ]
                    prompts.append(self.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    ))

                # Tokenize batch
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    # Suppress the padding warning since we've already configured it correctly
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*right-padding.*")
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=temperature,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.15,
                        )

                # Decode and parse
                for i in range(current_batch_size):
                    generated_tokens = outputs[i][inputs['input_ids'].shape[1]:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    # Parse response
                    question, answer = self._parse_problem(response)
                    if question and answer:
                        problems.append({
                            "question": question,
                            "answer": answer,
                            "raw_response": response
                        })

        return problems

    def _parse_problem(self, response: str) -> Tuple[str, str]:
        """Parse generated response to extract question and answer

        Args:
            response: Raw model output

        Returns:
            Tuple of (question, answer) or ("", "") if parsing fails
        """
        question = ""
        answer = ""

        # Try to extract from XML-like tags
        question_match = re.search(r'<question>\s*(.*?)\s*</question>', response, re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()

        # Extract answer
        answer_match = re.search(r'Answer:\s*([\d\.\-]+)', response)
        if answer_match:
            answer = answer_match.group(1).strip()

        # Fallback: extract first paragraph as question and last number as answer
        if not question:
            lines = response.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('Answer'):
                    question = line.strip()
                    break

        if not answer:
            numbers = re.findall(r'[\d\.\-]+', response)
            if numbers:
                answer = numbers[-1]

        return question, answer

    def compute_uncertainty_reward(
        self,
        problems: List[Dict],
        solver_responses: List[List[str]]
    ) -> List[float]:
        """Compute uncertainty reward based on solver's self-consistency

        Following R-Zero paper: r_uncertainty = 1 - 2|p̂(x; Sφ) - 0.5|

        Args:
            problems: List of generated problems
            solver_responses: For each problem, list of m solver responses

        Returns:
            List of uncertainty rewards
        """
        rewards = []

        for problem_idx, responses in enumerate(solver_responses):
            if not responses:
                rewards.append(0.0)
                continue

            # Extract answers and compute majority vote
            extracted_answers = []
            for resp in responses:
                # Extract numerical answer
                numbers = re.findall(r'[\d\.\-]+', resp)
                if numbers:
                    extracted_answers.append(numbers[-1])

            if not extracted_answers:
                rewards.append(0.0)
                continue

            # Compute empirical accuracy (self-consistency)
            from collections import Counter
            answer_counts = Counter(extracted_answers)
            most_common = answer_counts.most_common(1)[0]
            p_hat = most_common[1] / len(extracted_answers)

            # Uncertainty reward: maximized when p_hat = 0.5
            r_uncertainty = 1.0 - 2.0 * abs(p_hat - 0.5)
            rewards.append(r_uncertainty)

        return rewards

    def compute_repetition_penalty(
        self,
        problems: List[Dict],
        tau_bleu: float = 0.5
    ) -> List[float]:
        """Compute repetition penalty using BLEU-based clustering

        Args:
            problems: List of generated problems
            tau_bleu: BLEU threshold for clustering

        Returns:
            List of repetition penalties
        """
        if len(problems) <= 1:
            return [0.0] * len(problems)

        # Compute pairwise BLEU distances
        n = len(problems)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # Tokenize for BLEU
                ref = problems[i]["question"].split()
                hyp = problems[j]["question"].split()

                # Compute BLEU score with smoothing to avoid warnings
                try:
                    # Use smoothing to handle cases with no n-gram overlaps
                    smoothing = SmoothingFunction().method1
                    bleu = sentence_bleu([ref], hyp, smoothing_function=smoothing)
                except:
                    bleu = 0.0

                # Convert to distance
                distance = 1.0 - bleu
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tau_bleu,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Compute penalties based on cluster sizes
        penalties = []
        cluster_sizes = Counter(cluster_labels)

        for problem_idx in range(n):
            cluster_id = cluster_labels[problem_idx]
            cluster_size = cluster_sizes[cluster_id]
            # Penalty proportional to relative cluster size
            penalty = cluster_size / n
            penalties.append(penalty)

        return penalties

    def compute_format_penalty(self, problems: List[Dict]) -> List[float]:
        """Check if generated problems have proper format

        Args:
            problems: List of generated problems

        Returns:
            List of format penalties (0 if good, -1 if bad)
        """
        penalties = []

        for problem in problems:
            # Check basic format requirements
            if not problem.get("question") or not problem.get("answer"):
                penalties.append(-1.0)
            elif len(problem["question"]) < 20:  # Too short
                penalties.append(-0.5)
            else:
                penalties.append(0.0)

        return penalties

    def train_with_grpo(
        self,
        frozen_solver,
        num_problems: int = 100,
        max_steps: int = 5,
        m_solver_samples: int = 10
    ):
        """Train Challenger using GRPO with uncertainty reward

        Args:
            frozen_solver: Frozen solver model for computing uncertainty
            num_problems: Number of problems to generate per batch
            max_steps: Maximum GRPO training steps
            m_solver_samples: Number of solver samples for uncertainty computation
        """
        CLIFormatter.print_info("Training Challenger with GRPO...")

        # Generate training data prompts
        dataset_items = []
        for _ in range(num_problems):
            dataset_items.append({
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Generate a new math problem:"}
                ]
            })

        dataset = Dataset.from_list(dataset_items)

        # GRPO configuration (following paper)
        training_args = GRPOConfig(
            learning_rate=1e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=1e-2,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            logging_steps=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,  # Effective batch size = 8
            num_generations=4,  # G=4 from paper
            max_prompt_length=128,
            max_completion_length=200,
            max_steps=max_steps,
            save_steps=max_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=f"outputs/challenger_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Generation config
        generation_config = {
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Create custom reward function that uses frozen solver
        def compute_rewards(prompts, completions, **kwargs):
            """Compute composite rewards for generated problems"""
            batch_size = len(completions)

            # Parse problems from completions
            problems = []
            for completion in completions:
                response = completion[0]["content"]
                question, answer = self._parse_problem(response)
                problems.append({
                    "question": question,
                    "answer": answer,
                    "raw_response": response
                })

            # Get solver responses for uncertainty computation
            solver_responses = []
            for problem in problems:
                if problem["question"]:
                    # Get m samples from frozen solver
                    responses = []
                    for _ in range(m_solver_samples):
                        solver_output = frozen_solver.generate_solution(problem["question"])
                        responses.append(solver_output)
                    solver_responses.append(responses)
                else:
                    solver_responses.append([])

            # Compute individual reward components
            uncertainty_rewards = self.compute_uncertainty_reward(problems, solver_responses)
            repetition_penalties = self.compute_repetition_penalty(problems)
            format_penalties = self.compute_format_penalty(problems)

            # Composite reward
            composite_rewards = []
            for i in range(batch_size):
                reward = max(0, uncertainty_rewards[i] - repetition_penalties[i] + format_penalties[i])
                composite_rewards.append(reward)

            return composite_rewards

        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[compute_rewards],
            args=training_args,
            train_dataset=dataset,
            generation_config=generation_config,
        )

        # Custom logging
        original_log = trainer.log
        def formatted_log(logs, start_time=None):
            if logs and isinstance(logs, dict):
                step = trainer.state.global_step
                if step > 0 and 'reward' in logs:
                    CLIFormatter.format_grpo_stats(logs, step, max_steps)
                elif 'train_runtime' in logs:
                    runtime = logs.get('train_runtime', 0)
                    CLIFormatter.print_success(f"Challenger GRPO training completed in {runtime:.1f}s")

        trainer.log = formatted_log
        trainer.args.disable_tqdm = True

        # Train
        CLIFormatter.print_subheader("Challenger GRPO Training")
        CLIFormatter.print_info(f"Training for {max_steps} steps to learn problem generation...")
        print()
        trainer.train()
        print()

        # IMPORTANT: Set model back to eval mode after training
        self.model.eval()

    def save_checkpoint(self, iteration: int, run_id: str = None) -> str:
        """Save model checkpoint

        Args:
            iteration: Current iteration number
            run_id: Run identifier

        Returns:
            Path to saved checkpoint
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_dir = f"outputs/{run_id}/challenger/iteration_{iteration}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        CLIFormatter.print_info(f"Saving Challenger checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "problem_history_size": len(self.problem_history)
        }
        with open(f"{checkpoint_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return checkpoint_dir

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()