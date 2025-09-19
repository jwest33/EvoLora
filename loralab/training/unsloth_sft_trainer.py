"""Unsloth-optimized SFT trainer with TRL integration

Provides supervised fine-tuning with Unsloth optimizations and TRL's SFTTrainer
for improved performance and memory efficiency.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import torch
from tqdm import tqdm
from pathlib import Path
import os

try:
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only, standardize_data_formats
    TRL_AVAILABLE = True
    UNSLOTH_AVAILABLE = True
except ImportError as e:
    TRL_AVAILABLE = False
    UNSLOTH_AVAILABLE = False
    logging.warning(f"TRL or Unsloth not installed: {e}")

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("Datasets library not installed")

logger = logging.getLogger(__name__)


class UnslothSFTTrainer:
    """Optimized SFT trainer using Unsloth and TRL"""

    def __init__(self, model_manager, training_config: Dict[str, Any]):
        """Initialize the Unsloth SFT trainer

        Args:
            model_manager: UnslothModelManager or ModelManager instance
            training_config: Training configuration
        """
        if not TRL_AVAILABLE:
            raise ImportError("TRL is required. Install with: pip install trl")

        self.model_manager = model_manager
        self.training_config = training_config
        self.tokenizer = model_manager.get_tokenizer()

    def prepare_dataset(self, data: Union[List[Dict], Dataset],
                       format_type: str = "conversations") -> Dataset:
        """Prepare dataset for SFT training

        Args:
            data: Raw dataset (list of dicts or Dataset)
            format_type: Format type - "conversations", "instruct", or "completion"

        Returns:
            Formatted dataset ready for training
        """
        # Convert list to Dataset if needed
        if isinstance(data, list):
            if not DATASETS_AVAILABLE:
                raise ImportError("datasets library required. Install with: pip install datasets")

            # Determine format based on data structure
            if all('conversations' in item for item in data):
                dataset = Dataset.from_list(data)
                format_type = "conversations"
            elif all('instruction' in item or 'question' in item for item in data):
                # Convert to conversations format
                conversations = []
                for item in data:
                    conv = []

                    # Add system message if present
                    if 'system' in item:
                        conv.append({"role": "system", "content": item['system']})

                    # Add user message
                    user_content = item.get('instruction', item.get('question', ''))
                    if 'input' in item and item['input']:
                        user_content = f"{user_content}\n\nInput: {item['input']}"
                    conv.append({"role": "user", "content": user_content})

                    # Add assistant message
                    assistant_content = item.get('output', item.get('answer', item.get('response', '')))
                    conv.append({"role": "assistant", "content": assistant_content})

                    conversations.append({"conversations": conv})

                dataset = Dataset.from_list(conversations)
                format_type = "conversations"
            else:
                # Simple text format
                texts = []
                for item in data:
                    text = item.get('text', '')
                    if not text:
                        # Try to construct from question/answer
                        q = item.get('question', item.get('input', ''))
                        a = item.get('answer', item.get('output', ''))
                        text = f"Question: {q}\nAnswer: {a}"
                    texts.append({"text": text})
                dataset = Dataset.from_list(texts)
                format_type = "completion"
        else:
            dataset = data

        # Standardize format if Unsloth is available
        if UNSLOTH_AVAILABLE and format_type == "conversations":
            try:
                import platform
                # Force single process on Windows to avoid multiprocessing issues
                if platform.system() == 'Windows':
                    # Skip standardization on Windows due to multiprocessing issues
                    # The dataset will still work, just without Unsloth's optimizations
                    logger.debug("Skipping Unsloth standardization on Windows")
                else:
                    dataset = standardize_data_formats(dataset)
            except Exception as e:
                logger.warning(f"Could not standardize dataset format with Unsloth: {e}")

        # Apply chat template if using conversations
        if format_type == "conversations":
            def formatting_func(examples):
                convos = examples["conversations"]
                texts = []
                for convo in convos:
                    try:
                        text = self.tokenizer.apply_chat_template(
                            convo,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        texts.append(text)
                    except:
                        # Fallback to simple formatting
                        text = ""
                        for turn in convo:
                            role = turn.get("role", "")
                            content = turn.get("content", "")
                            text += f"{role}: {content}\n"
                        texts.append(text)
                return {"text": texts}

            dataset = dataset.map(
                formatting_func,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=1  # Force single process for Windows compatibility
            )

        return dataset

    def train(self, model: Any, train_data: Union[List[Dict], Dataset],
             learning_rate: float, epochs: int = 1,
             variant_id: str = "", val_data: Optional[Union[List[Dict], Dataset]] = None,
             train_on_responses: bool = False) -> Dict[str, float]:
        """Train model using TRL's SFTTrainer

        Args:
            model: Model with LoRA applied
            train_data: Training dataset
            learning_rate: Learning rate
            epochs: Number of training epochs
            variant_id: Identifier for logging
            val_data: Optional validation dataset
            train_on_responses: Only train on assistant responses

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting Unsloth SFT training for {variant_id}")

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = self.prepare_dataset(val_data) if val_data else None

        # Get training parameters
        batch_size = self.training_config.get('batch_size', 2)
        gradient_accumulation = self.training_config.get('gradient_accumulation_steps', 4)
        max_seq_length = self.training_config.get('max_seq_length', 2048)

        # Create SFT configuration
        sft_config = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size if eval_dataset else None,
            gradient_accumulation_steps=gradient_accumulation,
            warmup_ratio=self.training_config.get('warmup_ratio', 0.1),
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
            weight_decay=self.training_config.get('weight_decay', 0.01),
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"outputs/sft_{variant_id}",
            max_seq_length=max_seq_length,
            report_to="none",  # Disable wandb/tensorboard

            # Optional evaluation
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch" if eval_dataset else "no",
            save_total_limit=1,
            load_best_model_at_end=True if eval_dataset else False,

            # Memory optimizations
            fp16=False,  # Disabled for LoRA stability
            bf16=False,  # Disabled for LoRA stability
            gradient_checkpointing=False,  # Can conflict with LoRA
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

        # Apply response-only training if requested and Unsloth is available
        if train_on_responses and UNSLOTH_AVAILABLE:
            try:
                # Detect chat template markers
                chat_template = self.training_config.get('chat_template')

                if chat_template == "qwen3-instruct":
                    instruction_part = "<|im_start|>user\n"
                    response_part = "<|im_start|>assistant\n"
                elif chat_template == "llama3":
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n"
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n"
                elif chat_template == "gemma3":
                    instruction_part = "<start_of_turn>user\n"
                    response_part = "<start_of_turn>model\n"
                else:
                    # Generic markers
                    instruction_part = "User:"
                    response_part = "Assistant:"

                trainer = train_on_responses_only(
                    trainer,
                    instruction_part=instruction_part,
                    response_part=response_part
                )
                logger.info(f"Applied response-only training with markers: {instruction_part} / {response_part}")
            except Exception as e:
                logger.warning(f"Could not apply response-only training: {e}")
                
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        # Train
        train_output = trainer.train()

        # Extract metrics
        metrics = {
            'final_loss': train_output.metrics.get('train_loss', float('inf')),
            'total_steps': train_output.metrics.get('train_steps', 0),
            'samples_per_second': train_output.metrics.get('train_samples_per_second', 0),
        }

        # Add validation metrics if available
        if eval_dataset:
            eval_metrics = trainer.evaluate()
            metrics['eval_loss'] = eval_metrics.get('eval_loss', float('inf'))
            metrics['eval_perplexity'] = eval_metrics.get('eval_perplexity', float('inf'))

        logger.info(f"SFT training completed for {variant_id}")
        logger.info(f"  Final loss: {metrics['final_loss']:.4f}")
        if 'eval_loss' in metrics:
            logger.info(f"  Eval loss: {metrics['eval_loss']:.4f}")
        logger.info(f"  Samples/sec: {metrics['samples_per_second']:.2f}")

        # Clean up output directory to save space
        try:
            output_dir = Path(f"outputs/sft_{variant_id}")
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
        except:
            pass

        return metrics

    def train_with_early_stopping(self, model: Any, train_data: Union[List[Dict], Dataset],
                                 val_data: Union[List[Dict], Dataset],
                                 learning_rate: float, max_epochs: int = 10,
                                 patience: int = 3, variant_id: str = "") -> Dict[str, float]:
        """Train with early stopping based on validation loss

        Args:
            model: Model to train
            train_data: Training dataset
            val_data: Validation dataset
            learning_rate: Learning rate
            max_epochs: Maximum number of epochs
            patience: Number of epochs to wait for improvement
            variant_id: Identifier for logging

        Returns:
            Dictionary of training metrics
        """
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_metrics = {}

        for epoch in range(max_epochs):
            # Train for one epoch
            metrics = self.train(
                model=model,
                train_data=train_data,
                learning_rate=learning_rate,
                epochs=1,
                variant_id=f"{variant_id}_epoch{epoch+1}",
                val_data=val_data,
                train_on_responses=self.training_config.get('train_on_completions_only', False)
            )

            val_loss = metrics.get('eval_loss', float('inf'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                epochs_without_improvement = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return best_metrics


def create_trainer(model_manager, training_config: Dict[str, Any]) -> Any:
    """Factory function to create appropriate trainer

    Args:
        model_manager: Model manager instance
        training_config: Training configuration

    Returns:
        Appropriate trainer based on configuration
    """
    method = training_config.get('method', 'sft')
    use_trl = training_config.get('use_trl_trainer', False)

    # Use Unsloth SFT trainer if TRL is requested and available
    if method == 'sft' and use_trl and TRL_AVAILABLE:
        return UnslothSFTTrainer(model_manager, training_config)
    elif method == 'grpo':
        from .grpo_trainer import GRPOTrainer
        return GRPOTrainer(model_manager, training_config)
    else:
        # Fall back to original self-supervised trainer
        from .self_supervised import SelfSupervisedTrainer
        return SelfSupervisedTrainer(model_manager, training_config)