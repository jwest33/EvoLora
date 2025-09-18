"""Self-supervised training module for LoRA adapters

Trains LoRA adapters directly on labeled datasets without a teacher model.
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple dataset for text data"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """Initialize dataset

        Args:
            data: List of dictionaries with 'question' and 'answer' fields
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format text with question and answer
        question = item.get('question', item.get('text', ''))
        answer = item.get('answer', item.get('label', ''))

        # Create full text
        text = f"Question: {question}\nAnswer: {answer}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For causal LM, labels = input_ids
        }


class SelfSupervisedTrainer:
    """Trainer for self-supervised LoRA training"""

    def __init__(self, model_manager, training_config: Dict[str, Any]):
        """Initialize trainer

        Args:
            model_manager: ModelManager instance
            training_config: Training configuration dictionary
        """
        self.model_manager = model_manager
        self.training_config = training_config
        self.tokenizer = model_manager.get_tokenizer()

    def train(self,
             model: Any,
             train_data: List[Dict],
             learning_rate: float,
             epochs: int = 1,
             variant_id: str = "") -> float:
        """Train a model on the dataset

        Args:
            model: Model to train (with LoRA applied)
            train_data: Training dataset
            learning_rate: Learning rate to use
            epochs: Number of epochs to train
            variant_id: Identifier for logging

        Returns:
            Average training loss
        """
        logger.info(f"Starting training for {variant_id}")
        device = next(model.parameters()).device

        # Create dataset and dataloader
        dataset = TextDataset(train_data, self.tokenizer)

        # Determine batch size - use smaller if memory constrained
        batch_size = min(
            self.training_config.get('batch_size', 4),
            4  # Cap at 4 for memory safety
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )

        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )

        # Gradient accumulation settings
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        max_grad_norm = self.training_config.get('max_grad_norm', 1.0)

        # Training loop
        model.train()
        total_loss = 0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_batches = 0

            progress = tqdm(dataloader, desc=f"Training {variant_id} - Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(progress):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Update weights if accumulation complete
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                max_grad_norm
                            )

                        optimizer.step()
                        optimizer.zero_grad()

                    # Track losses
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    epoch_batches += 1

                    # Update progress bar
                    progress.set_postfix({
                        'loss': loss.item() * gradient_accumulation_steps,
                        'avg_loss': epoch_loss / epoch_batches
                    })

                    # Periodic memory cleanup
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"OOM error for {variant_id}! Skipping batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

            # Log epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
            logger.info(f"{variant_id} - Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")

            total_loss += epoch_loss
            num_batches += epoch_batches

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        return avg_loss

    def train_with_validation(self,
                             model: Any,
                             train_data: List[Dict],
                             val_data: List[Dict],
                             learning_rate: float,
                             epochs: int = 1,
                             variant_id: str = "") -> Dict[str, float]:
        """Train with validation monitoring

        Args:
            model: Model to train
            train_data: Training dataset
            val_data: Validation dataset
            learning_rate: Learning rate
            epochs: Number of epochs
            variant_id: Identifier for logging

        Returns:
            Dictionary with training and validation metrics
        """
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            avg_train_loss = self.train(
                model=model,
                train_data=train_data,
                learning_rate=learning_rate,
                epochs=1,
                variant_id=f"{variant_id}_epoch{epoch+1}"
            )
            train_losses.append(avg_train_loss)

            # Validation
            val_loss = self.validate(model, val_data, variant_id)
            val_losses.append(val_loss)

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")

            # Early stopping check
            if self._should_early_stop(val_losses):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return {
            'final_train_loss': train_losses[-1] if train_losses else float('inf'),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

    def validate(self, model: Any, val_data: List[Dict], variant_id: str = "") -> float:
        """Validate model on validation set

        Args:
            model: Model to validate
            val_data: Validation dataset
            variant_id: Identifier for logging

        Returns:
            Average validation loss
        """
        model.eval()
        device = next(model.parameters()).device

        dataset = TextDataset(val_data, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.get('batch_size', 4),
            shuffle=False
        )

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validating {variant_id}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        model.train()  # Set back to training mode
        return avg_loss

    def _should_early_stop(self, val_losses: List[float], patience: int = 3) -> bool:
        """Check if training should stop early

        Args:
            val_losses: List of validation losses
            patience: Number of epochs to wait for improvement

        Returns:
            True if should stop early
        """
        if len(val_losses) < patience + 1:
            return False

        # Check if validation loss hasn't improved in patience epochs
        recent_losses = val_losses[-patience:]
        best_recent = min(recent_losses)
        previous_best = min(val_losses[:-patience])

        return best_recent >= previous_best
