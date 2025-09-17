"""
LoRA Solver training module with GRPO (Group Relative Policy Optimization).
Handles fine-tuning of the solver model on challenger-generated tasks.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example for GRPO."""
    input_text: str
    target_text: str
    reward: float
    task_id: str


class SolverDataset(Dataset):
    """Dataset for solver training."""

    def __init__(self, examples: List[TrainingExample], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Combine input and target for causal LM
        full_text = f"{example.input_text}\n{example.target_text}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'reward': torch.tensor(example.reward, dtype=torch.float32),
            'task_id': example.task_id
        }


class LoRASolverTrainer:
    """Trains solver model using GRPO."""

    def __init__(self,
                 solver_model,
                 training_config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            solver_model: SolverModel instance
            training_config: Training configuration
        """
        self.solver = solver_model
        self.config = training_config

        # GRPO parameters - ensure they're numeric
        self.learning_rate = float(training_config.get('learning_rate', 1e-5))
        self.batch_size = int(training_config.get('batch_size', 8))
        self.num_rollouts = int(training_config.get('num_rollouts', 4))
        self.kl_penalty = float(training_config.get('kl_penalty', 0.1))
        self.clip_ratio = float(training_config.get('clip_ratio', 0.2))
        self.max_grad_norm = float(training_config.get('max_grad_norm', 1.0))

        # Initialize optimizer - only for trainable parameters
        trainable_params = [p for p in self.solver.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")

        logger.info(f"Found {len(trainable_params)} trainable parameter tensors")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Reference model for KL penalty (frozen copy)
        self.reference_model = None
        self._create_reference_model()

    def _create_reference_model(self):
        """Create frozen reference model for KL divergence."""
        logger.info("Creating reference model for KL penalty")
        # For now, we'll skip the reference model to avoid PEFT issues
        # The KL penalty will be disabled
        self.reference_model = None
        return

        # TODO: Fix reference model creation for KL divergence
        # The issue is with PEFT's adapter management changing in newer versions

    def train_iteration(self,
                       tasks: List,
                       pseudo_labels: List[str],
                       challenger_scores: List[float]) -> Dict[str, float]:
        """
        Run one training iteration with GRPO.

        Args:
            tasks: List of tasks
            pseudo_labels: Challenger-generated labels
            challenger_scores: Quality scores from challenger

        Returns:
            Training metrics
        """
        # Generate rollouts for each task
        all_examples = []
        all_rewards = []

        print(f"\n[Step 5/5] Training Solver with GRPO...")
        print(f"           Generating {self.num_rollouts} rollouts per task...")
        for i, (task, label, base_score) in enumerate(zip(tasks, pseudo_labels, challenger_scores)):
            if i % 20 == 0 and i > 0:
                print(f"           Processed {i}/{len(tasks)} tasks...")
            examples, rewards = self._generate_rollouts(task, label, base_score)
            all_examples.extend(examples)
            all_rewards.extend(rewards)

        # Normalize rewards per group (GRPO)
        normalized_rewards = self._normalize_rewards(all_rewards, self.num_rollouts)

        # Create training examples
        training_examples = []
        for ex, reward in zip(all_examples, normalized_rewards):
            training_examples.append(
                TrainingExample(
                    input_text=ex['input'],
                    target_text=ex['output'],
                    reward=reward,
                    task_id=ex['task_id']
                )
            )

        # Create dataset and dataloader
        dataset = SolverDataset(training_examples, self.solver.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Training loop
        return self._train_epoch(dataloader)

    def _generate_rollouts(self,
                          task,
                          pseudo_label: str,
                          base_score: float) -> Tuple[List, List]:
        """
        Generate multiple rollouts for a single task.

        Args:
            task: Task object
            pseudo_label: Target label
            base_score: Base quality score

        Returns:
            Tuple of (examples, rewards)
        """
        examples = []
        rewards = []

        # Create prompt based on task type
        if task.metadata['task_type'] == 'code_documentation':
            prompt = f"Generate documentation for this code:\n\n{task.input_text}\n\nDocumentation:"
        else:
            prompt = f"Task: {task.input_text}\n\nResponse:"

        # Generate multiple completions
        completions = self.solver.generate(
            prompt,
            max_tokens=256,
            temperature=0.8,
            num_return=self.num_rollouts
        )

        for completion in completions:
            # Calculate reward based on similarity to pseudo-label
            reward = self._calculate_reward(completion, pseudo_label, base_score)

            examples.append({
                'input': prompt,
                'output': completion,
                'task_id': task.id
            })
            rewards.append(reward)

        return examples, rewards

    def _calculate_reward(self,
                         generated: str,
                         target: str,
                         base_score: float) -> float:
        """
        Calculate reward for generated output.

        Args:
            generated: Generated text
            target: Target pseudo-label
            base_score: Base quality score from challenger

        Returns:
            Reward value
        """
        # Simple reward based on length ratio and base score
        len_ratio = min(len(generated), len(target)) / max(len(generated), len(target))

        # Token overlap (simplified)
        gen_tokens = set(generated.lower().split())
        target_tokens = set(target.lower().split())
        overlap = len(gen_tokens & target_tokens) / max(len(gen_tokens), len(target_tokens))

        # Combine metrics
        reward = (len_ratio * 0.3 + overlap * 0.3 + base_score * 0.4)

        return reward

    def _normalize_rewards(self, rewards: List[float], group_size: int) -> List[float]:
        """
        Normalize rewards per group (GRPO style).

        Args:
            rewards: Flat list of rewards
            group_size: Size of each group

        Returns:
            Normalized rewards
        """
        normalized = []
        num_groups = len(rewards) // group_size

        for i in range(num_groups):
            group = rewards[i * group_size:(i + 1) * group_size]
            mean = np.mean(group)
            std = np.std(group) + 1e-8  # Add epsilon for stability

            for r in group:
                normalized.append((r - mean) / std)

        return normalized

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch with GRPO.

        Args:
            dataloader: Training data loader

        Returns:
            Training metrics
        """
        self.solver.model.train()

        # Ensure gradients are enabled for LoRA parameters
        for name, param in self.solver.model.named_parameters():
            if param.requires_grad:
                param.requires_grad_(True)
                logger.debug(f"Gradient enabled for: {name}")

        total_loss = 0
        total_policy_loss = 0
        total_kl_loss = 0
        num_batches = 0

        progress = tqdm(dataloader, desc="           Training LoRA adapter")
        for batch in progress:
            # Move to device
            input_ids = batch['input_ids'].to(self.solver.device)
            attention_mask = batch['attention_mask'].to(self.solver.device)
            rewards = batch['reward'].to(self.solver.device)

            # Forward pass
            outputs = self.solver.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Get logits and calculate policy loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Calculate log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Apply rewards (advantages in GRPO)
            policy_loss = -(gathered_log_probs * rewards.unsqueeze(-1)).mean()

            # KL divergence penalty
            if self.reference_model is not None:
                kl_loss = self._compute_kl_penalty(
                    input_ids,
                    attention_mask,
                    logits
                )
            else:
                kl_loss = torch.tensor(0.0, device=self.solver.device, requires_grad=True)

            # Total loss
            loss = policy_loss + self.kl_penalty * kl_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.solver.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            if isinstance(kl_loss, torch.Tensor):
                total_kl_loss += kl_loss.item()
            num_batches += 1

            # Update progress bar
            progress.set_postfix({
                'loss': total_loss / num_batches,
                'policy': total_policy_loss / num_batches,
                'kl': total_kl_loss / num_batches if total_kl_loss > 0 else 0
            })

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches if total_kl_loss > 0 else 0
        }

    def _compute_kl_penalty(self,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           current_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from reference model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            current_logits: Current model logits

        Returns:
            KL divergence loss
        """
        # Get reference logits
        self.solver.set_adapter("reference")
        with torch.no_grad():
            ref_outputs = self.solver.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits

        # Switch back to training adapter
        self.solver.set_adapter("default")

        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(current_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction='batchmean'
        )

        return kl_div

    def evaluate(self, eval_tasks: List, eval_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate solver on validation tasks.

        Args:
            eval_tasks: Evaluation tasks
            eval_labels: Target labels

        Returns:
            Evaluation metrics
        """
        self.solver.model.eval()
        total_score = 0
        num_tasks = len(eval_tasks)

        print(f"           Evaluating on {num_tasks} validation tasks...")

        with torch.no_grad():
            for i, (task, label) in enumerate(zip(eval_tasks, eval_labels)):
                if i % 10 == 0 and i > 0:
                    print(f"           Evaluated {i}/{num_tasks} tasks...")
                # Generate response
                if task.metadata['task_type'] == 'code_documentation':
                    prompt = f"Generate documentation for this code:\n\n{task.input_text}\n\nDocumentation:"
                else:
                    prompt = f"Task: {task.input_text}\n\nResponse:"

                response = self.solver.generate(
                    prompt,
                    max_tokens=256,
                    temperature=0.3,
                    num_return=1
                )[0]

                # Calculate score
                score = self._calculate_reward(response, label, 1.0)
                total_score += score

        return {
            'eval_score': total_score / num_tasks if num_tasks > 0 else 0
        }