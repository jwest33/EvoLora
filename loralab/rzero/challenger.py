"""Challenger Agent for R-Zero Framework

The Challenger generates increasingly difficult questions to push the Solver
to the edge of its capabilities, implementing the uncertainty-based reward
mechanism from the R-Zero paper.
"""

import logging
import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


@dataclass
class ChallengerQuestion:
    """Container for a generated question with metadata"""
    text: str
    difficulty_estimate: float
    iteration: int
    reward: float = 0.0
    solver_accuracy: float = 0.0


class ChallengerAgent:
    """Challenger agent that generates difficult math problems"""

    def __init__(self, model_manager, config, tokenizer=None):
        """Initialize the Challenger agent

        Args:
            model_manager: UnslothModelManager for model handling
            config: ChallengerConfig with hyperparameters
            tokenizer: Optional tokenizer (uses model_manager's if not provided)
        """
        self.model_manager = model_manager
        self.config = config
        self.tokenizer = tokenizer or model_manager.get_tokenizer()
        self.current_iteration = 0
        self.generated_questions = []

        # Compile regex patterns for format checking
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for question validation"""
        # Pattern to match question tags
        self.question_pattern = re.compile(
            r"<question>(.*?)</question>",
            flags=re.MULTILINE | re.DOTALL
        )

        # Pattern to match answer in boxed format
        self.answer_pattern = re.compile(
            r"\\boxed\{([^}]+)\}",
            flags=re.MULTILINE
        )

        # Pattern to extract numbers
        self.number_pattern = re.compile(
            r"[-+]?(?:\d*\.\d+|\d+\.?\d*)"
        )

    def generate_questions(self, num_questions: int, solver_model=None) -> List[ChallengerQuestion]:
        """Generate challenging questions using GRPO

        Args:
            num_questions: Number of questions to generate
            solver_model: Current solver model for uncertainty calculation

        Returns:
            List of generated questions
        """
        logger.info(f"Generating {num_questions} questions for iteration {self.current_iteration}")

        # Create prompts for question generation
        prompts = self._create_generation_prompts(num_questions)

        # Generate questions using the model
        generated_texts = self._generate_with_model(prompts)

        # Calculate rewards if solver is provided
        questions = []
        for i, text in enumerate(generated_texts):
            question = ChallengerQuestion(
                text=text,
                difficulty_estimate=0.5 + 0.1 * self.current_iteration,  # Progressive difficulty
                iteration=self.current_iteration
            )

            # Calculate uncertainty reward if solver is available
            if solver_model:
                question.reward, question.solver_accuracy = self._calculate_uncertainty_reward(
                    question.text, solver_model
                )

            questions.append(question)

        self.generated_questions.extend(questions)
        return questions

    def _create_generation_prompts(self, num_prompts: int) -> List[Dict]:
        """Create prompts for question generation"""
        prompts = []

        system_message = """You are an expert competition-math problem setter. FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. The problem should be a word problem involving real-world scenarios.

Focus on creating problems that:
- Involve multi-step reasoning
- Combine different mathematical operations
- Use realistic scenarios (shopping, travel, work, school, etc.)
- Require careful reading to understand what's being asked

THEN, output exactly the following two blocks:

<question>
{The full problem statement on one or more lines}
</question>

\\boxed{final numeric answer}

Do NOT output anything else—no explanations, no extra markup."""

        user_message = f"""Generate a new challenging math word problem at difficulty level {self.current_iteration + 1}/10.

The problem should be solvable by a strong high school student but require careful thinking.
Make it a realistic scenario that someone might encounter in daily life.

Remember to format the output exactly as instructed."""

        for _ in range(num_prompts):
            prompts.append({
                "prompt": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            })

        return prompts

    def _generate_with_model(self, prompts: List[Dict]) -> List[str]:
        """Generate text using the Challenger model"""
        model = self.model_manager.get_model()
        generated_texts = []

        for prompt_dict in prompts:
            # Format prompt for generation
            messages = prompt_dict["prompt"]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate with specified parameters
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and extract generated text
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt part
            generated = generated[len(formatted_prompt):]
            generated_texts.append(generated.strip())

        return generated_texts

    def _calculate_uncertainty_reward(
        self, question_text: str, solver_model
    ) -> Tuple[float, float]:
        """Calculate uncertainty-based reward for a question

        Args:
            question_text: The generated question
            solver_model: The current solver model

        Returns:
            Tuple of (reward, solver_accuracy)
        """
        # Extract question from tags if present
        question_match = self.question_pattern.search(question_text)
        if question_match:
            question = question_match.group(1).strip()
        else:
            question = question_text

        # Generate multiple solver responses
        solver_responses = []
        for _ in range(self.config.num_solver_responses):
            response = self._get_solver_response(question, solver_model)
            solver_responses.append(response)

        # Extract answers and calculate accuracy
        answers = []
        for response in solver_responses:
            answer = self._extract_answer(response)
            if answer:
                answers.append(answer)

        if not answers:
            # No valid answers, maximum uncertainty
            return 1.0, 0.0

        # Calculate pseudo-label via majority vote
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        # Calculate empirical accuracy p̂
        accuracy = count / len(solver_responses)

        # Calculate uncertainty reward: r = 1 - 2|p̂ - 0.5|
        uncertainty_reward = 1.0 - 2 * abs(accuracy - 0.5)

        return uncertainty_reward, accuracy

    def _get_solver_response(self, question: str, solver_model) -> str:
        """Get a single response from the solver model"""
        # Format prompt for solver
        solver_prompt = f"""Solve the following math problem step by step.

Problem: {question}

Solution:"""

        inputs = self.tokenizer(solver_prompt, return_tensors="pt")
        inputs = {k: v.to(solver_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = solver_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for solver
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(solver_prompt):]

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract numeric answer from solver response"""
        # Look for Final Answer pattern
        final_match = re.search(r"Final Answer:\s*([-\d,.]+)", response, re.IGNORECASE)
        if final_match:
            return final_match.group(1).replace(",", "")

        # Look for answer patterns
        answer_patterns = [
            r"answer[:\s]+\$?([-\d,.]+)",
            r"total[:\s]+\$?([-\d,.]+)",
            r"result[:\s]+\$?([-\d,.]+)",
            r"=\s*([-\d,.]+)\s*$",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")

        # Last resort: find last number
        numbers = re.findall(r"[-\d,.]+", response)
        if numbers:
            return numbers[-1].replace(",", "")

        return None

    def calculate_composite_reward(
        self, questions: List[str], solver_model=None, batch_id: int = 0
    ) -> np.ndarray:
        """Calculate composite reward including uncertainty and repetition penalty

        Args:
            questions: List of generated question texts
            solver_model: Current solver model
            batch_id: Batch identifier for logging

        Returns:
            Array of composite rewards
        """
        rewards = np.zeros(len(questions))

        # 1. Format check penalty
        for i, q in enumerate(questions):
            if not self.question_pattern.search(q) or not self.answer_pattern.search(q):
                rewards[i] = 0.0  # Invalid format gets 0 reward
                continue

            # 2. Uncertainty reward
            if solver_model:
                uncertainty_reward, accuracy = self._calculate_uncertainty_reward(q, solver_model)
                rewards[i] += uncertainty_reward * 2.0  # Weight factor

            # 3. Length reward (encourage substantial problems)
            question_match = self.question_pattern.search(q)
            if question_match:
                question_text = question_match.group(1)
                word_count = len(question_text.split())
                if word_count > 30:
                    rewards[i] += 0.5
                elif word_count < 10:
                    rewards[i] -= 0.5

        # 4. Repetition penalty (applied at batch level)
        rep_penalties = self._calculate_repetition_penalty(questions)
        rewards -= rep_penalties

        # Ensure non-negative rewards
        rewards = np.maximum(rewards, 0.0)

        # Log sample rewards for debugging
        if batch_id == 0 and len(questions) > 0:
            logger.info(f"Sample rewards (first 5): {rewards[:5]}")
            logger.info(f"Average reward: {np.mean(rewards):.4f}")

        return rewards

    def _calculate_repetition_penalty(self, questions: List[str]) -> np.ndarray:
        """Calculate repetition penalty using BLEU similarity clustering"""
        n = len(questions)
        if n <= 1:
            return np.zeros(n)

        # Extract question texts
        question_texts = []
        for q in questions:
            match = self.question_pattern.search(q)
            if match:
                question_texts.append(match.group(1))
            else:
                question_texts.append(q)

        # Calculate pairwise BLEU distances
        distances = np.zeros((n, n))
        smoother = SmoothingFunction()

        for i in range(n):
            for j in range(i + 1, n):
                # Tokenize by words
                ref = question_texts[i].split()
                hyp = question_texts[j].split()

                # Calculate BLEU score
                try:
                    bleu = sentence_bleu(
                        [ref], hyp,
                        smoothing_function=smoother.method1
                    )
                    # Convert to distance
                    distance = 1.0 - bleu
                except:
                    distance = 1.0

                distances[i, j] = distance
                distances[j, i] = distance

        # Perform clustering
        if n > 2:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.config.bleu_threshold,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distances)
        else:
            # For 2 questions, simple threshold check
            cluster_labels = [0, 1] if distances[0, 1] > self.config.bleu_threshold else [0, 0]

        # Calculate penalties based on cluster sizes
        penalties = np.zeros(n)
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_size = np.sum(cluster_mask)

            if cluster_size > 1:
                # Penalty proportional to cluster size
                penalty = self.config.repetition_penalty_weight * (cluster_size / n)
                penalties[cluster_mask] = penalty

        return penalties

    def train_with_grpo(self, solver_model, num_steps: int = None):
        """Train the Challenger using GRPO with uncertainty rewards

        Args:
            solver_model: Current solver model for uncertainty calculation
            num_steps: Number of training steps (uses config default if None)
        """
        from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig

        num_steps = num_steps or self.config.max_steps
        logger.info(f"Training Challenger with GRPO for {num_steps} steps")

        # Prepare training configuration
        # Ensure batch size is at least 1
        batch_size = max(1, self.config.batch_size)

        grpo_config = GRPOConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_steps=num_steps,
            per_device_train_batch_size=batch_size,  # Use full batch size, not divided
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations_per_prompt,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            optim="adamw_8bit",
            logging_steps=1,
            report_to="none",
            output_dir=f"outputs/challenger_iter_{self.current_iteration}",
        )

        # Create reward function for GRPO
        def reward_func(completions, **kwargs):
            """Reward function for GRPO training"""
            # Extract text from completions
            texts = []
            for comp in completions:
                if isinstance(comp, list) and len(comp) > 0:
                    texts.append(comp[0].get('content', ''))
                else:
                    texts.append(str(comp))

            # Calculate composite rewards
            rewards = self.calculate_composite_reward(texts, solver_model)
            return rewards.tolist()

        # Create training dataset
        train_prompts = self._create_generation_prompts(
            self.config.num_questions_per_iteration // self.config.num_generations_per_prompt
        )

        # Initialize GRPO trainer
        model = self.model_manager.get_model()
        trainer = TRLGRPOTrainer(
            model=model,
            processing_class=self.tokenizer,
            reward_funcs=[reward_func],
            args=grpo_config,
            train_dataset=train_prompts,
        )

        # Train
        train_output = trainer.train()

        # Update iteration counter
        self.current_iteration += 1

        logger.info(f"Challenger training completed. Loss: {train_output.metrics.get('train_loss', 0):.4f}")

        return train_output.metrics

    def save_checkpoint(self, path: str):
        """Save Challenger checkpoint"""
        import os
        import json

        os.makedirs(path, exist_ok=True)

        # Save model
        model = self.model_manager.get_model()
        self.model_manager.save_model(model, os.path.join(path, "model"))

        # Save metadata
        metadata = {
            "iteration": self.current_iteration,
            "num_questions_generated": len(self.generated_questions),
            "config": self.config.__dict__,
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Challenger checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load Challenger checkpoint"""
        import os
        import json

        # Load model - recreate with LoRA from saved weights
        # For now, just log that loading is not fully implemented
        # In production, you would reload the base model and apply the saved LoRA weights
        logger.info(f"Note: Model checkpoint loading not fully implemented yet")

        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.current_iteration = metadata["iteration"]
        logger.info(f"Challenger checkpoint loaded from {path}")
