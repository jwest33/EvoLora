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
from ..utils.memory_monitor import MemoryMonitor
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Optimized dataset with pre-tokenization support"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256,
                 pre_tokenize: bool = True, cache_dir: str = None):
        """Initialize dataset

        Args:
            data: List of dictionaries with 'question' and 'answer' fields
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            pre_tokenize: Whether to pre-tokenize all data
            cache_dir: Directory to cache tokenized data
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = None

        if pre_tokenize:
            self._pre_tokenize_data(cache_dir)

    def _pre_tokenize_data(self, cache_dir):
        """Pre-tokenize all data for faster loading"""
        import os
        import pickle
        import hashlib

        # Create cache key from data characteristics
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = hashlib.md5(
                f"{len(self.data)}_{self.max_length}_{self.tokenizer.name_or_path}".encode()
            ).hexdigest()
            cache_path = os.path.join(cache_dir, f"tokenized_{cache_key}.pkl")

            # Try loading from cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        self.tokenized_data = pickle.load(f)
                    logger.info(f"Loaded pre-tokenized data from cache: {cache_path}")
                    return
                except:
                    pass

        logger.info("Pre-tokenizing dataset...")
        self.tokenized_data = []

        # Batch tokenization for efficiency
        texts = []
        for item in self.data:
            question = item.get('question', item.get('text', ''))
            answer = item.get('answer', item.get('label', ''))
            texts.append(f"Question: {question}\nAnswer: {answer}")

        # Tokenize in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            for j in range(len(batch_texts)):
                self.tokenized_data.append({
                    'input_ids': batch_encoding['input_ids'][j],
                    'attention_mask': batch_encoding['attention_mask'][j],
                    'labels': batch_encoding['input_ids'][j]
                })

        # Save to cache
        if cache_dir and cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.tokenized_data, f)
                logger.info(f"Saved pre-tokenized data to cache: {cache_path}")
            except:
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.tokenized_data:
            return self.tokenized_data[idx]

        # Fallback to on-the-fly tokenization
        item = self.data[idx]
        question = item.get('question', item.get('text', ''))
        answer = item.get('answer', item.get('label', ''))
        text = f"Question: {question}\nAnswer: {answer}"

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
            'labels': encoding['input_ids'].squeeze()
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
             variant_id: str = "",
             use_amp: bool = True,
             monitor_memory: bool = True,
             job_tracker: Any = None,
             **kwargs) -> float:
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
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        logger.info(f"Starting training for {variant_id}")
        device = next(model.parameters()).device

        # Create dataset and dataloader with optimizations
        max_length = self.training_config.get('max_seq_length', 256)
        cache_dir = self.training_config.get('cache_dir', 'cache')

        dataset = TextDataset(
            train_data,
            self.tokenizer,
            max_length=max_length,
            pre_tokenize=True,
            cache_dir=cache_dir
        )

        # Use configured batch size
        batch_size = self.training_config.get('batch_size', 16)
        num_workers = self.training_config.get('dataloader_num_workers', 2)
        pin_memory = self.training_config.get('dataloader_pin_memory', True)
        prefetch_factor = self.training_config.get('dataloader_prefetch_factor', 2)

        # Windows-specific adjustments
        import platform
        if platform.system() == 'Windows':
            # Check if using Unsloth - force single-threaded to avoid pickling issues
            if hasattr(self.model_manager, '__class__') and 'Unsloth' in self.model_manager.__class__.__name__:
                num_workers = 0  # Disable multiprocessing for Unsloth on Windows
                pin_memory = False  # Also disable pinned memory
                logger.debug("Disabled dataloader multiprocessing for Unsloth on Windows")
            else:
                num_workers = min(num_workers, 2)  # Limit workers on Windows

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )

        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )

        # Gradient accumulation settings
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 4)
        max_grad_norm = self.training_config.get('max_grad_norm', 1.0)

        # Setup mixed precision training
        scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
        use_bf16 = self.training_config.get('bf16', True)
        use_fp16 = self.training_config.get('fp16', False) and not use_bf16

        # Enable gradient checkpointing if configured
        if self.training_config.get('gradient_checkpointing', False):
            # Only enable if model supports it and not using LoRA
            # LoRA + gradient checkpointing can cause issues with gradients
            if hasattr(model, 'gradient_checkpointing_enable'):
                try:
                    model.gradient_checkpointing_enable()
                    logger.info("Enabled gradient checkpointing")
                except Exception as e:
                    logger.warning(f"Could not enable gradient checkpointing: {e}")

        # Initialize memory monitor if requested
        memory_monitor = MemoryMonitor(log_interval=10) if monitor_memory else None
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
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

                # Forward pass with mixed precision
                try:
                    # For LoRA training, we'll use regular precision to avoid gradient issues
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss / gradient_accumulation_steps

                    # Backward pass
                    if scaler and use_fp16:
                        # Scale loss for mixed precision
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()
                    else:
                        loss.backward()

                    # Update weights if accumulation complete
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        if scaler and use_fp16:
                            # Unscale gradients before clipping
                            scaler.unscale_(optimizer)

                            # Gradient clipping
                            if max_grad_norm:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_grad_norm
                                )

                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Gradient clipping
                            if max_grad_norm:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_grad_norm
                                )

                            optimizer.step()

                        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                    # Track losses
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    epoch_batches += 1

                    # Update progress bar
                    progress.set_postfix({
                        'loss': loss.item() * gradient_accumulation_steps,
                        'avg_loss': epoch_loss / epoch_batches
                    })

                    # Monitor memory if enabled
                    if memory_monitor:
                        memory_monitor.log_memory_status(batch_idx, variant_id)

                    # More frequent memory cleanup to reduce fragmentation
                    if batch_idx % 20 == 0 and batch_idx > 0 and torch.cuda.is_available():
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

            # Update job tracker if available
            if job_tracker:
                job_tracker.complete_epoch(
                    epoch_num=epoch + 1,
                    metrics={'loss': avg_epoch_loss}
                )

            total_loss += epoch_loss
            num_batches += epoch_batches

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Print memory summary if monitoring was enabled
        if memory_monitor:
            logger.info(memory_monitor.get_summary())

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
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
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
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
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
