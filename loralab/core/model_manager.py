"""Single model manager for self-supervised LoRA training

Manages the base model and creates LoRA variants for evolution.
"""

import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

try:
    from ..utils.cli_formatter import CLIFormatter
    USE_FORMATTER = True
except ImportError:
    USE_FORMATTER = False

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the base model and tokenizer for LoRA training"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with model configuration

        Args:
            config: Model configuration dictionary containing:
                - path: HuggingFace model ID or local path
                - torch_dtype: Data type for model weights
                - device_map: Device placement strategy
                - trust_remote_code: Whether to trust remote code
        """
        self.config = config
        self.base_model = None
        self.tokenizer = None
        self._device = None

    def load_base_model(self):
        """Load the base model and tokenizer"""
        model_path = self.config['path']
        logger.info(f"Loading base model: {model_path}")

        try:
            # Determine device
            device_map = self.config.get('device_map', 'cpu')

            # Check if user wants CUDA but it's not available
            if device_map != 'cpu':
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device_map = 'cpu'
                else:
                    logger.info(f"CUDA available, using device: {device_map}")

            # Load model with memory optimization
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, self.config.get('torch_dtype', 'float16')),
                device_map=device_map,
                trust_remote_code=self.config.get('trust_remote_code', True),
                low_cpu_mem_usage=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.get('trust_remote_code', True),
                padding_side="left"
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Get device
            self._device = next(self.base_model.parameters()).device

            if USE_FORMATTER:
                CLIFormatter.print_success(f"Model loaded successfully on {self._device}")
            else:
                logger.info(f"Model loaded successfully on {self._device}")

            # Print model info
            total_params = sum(p.numel() for p in self.base_model.parameters())
            if USE_FORMATTER:
                CLIFormatter.print_info(f"Total parameters: {total_params:,}")
            else:
                logger.info(f"Total parameters: {total_params:,}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_lora_variant(self, lora_config: Dict[str, Any]) -> Any:
        """Create a new LoRA variant with specified configuration

        Args:
            lora_config: LoRA configuration containing:
                - rank: LoRA rank
                - alpha: LoRA alpha (scaling factor)
                - dropout: Dropout rate
                - target_modules: List of modules to apply LoRA to

        Returns:
            PEFT model with LoRA applied
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model() first.")

        # Create LoRA configuration
        peft_config = LoraConfig(
            r=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            target_modules=lora_config['target_modules'],
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )

        # Apply LoRA to base model (creates a new PEFT model)
        lora_model = get_peft_model(self.base_model, peft_config)

        # Ensure model is in training mode
        lora_model.train()

        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in lora_model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        logger.debug(f"LoRA variant created: rank={lora_config['rank']}, "
                    f"trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")

        return lora_model

    def get_tokenizer(self):
        """Get the tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_base_model() first.")
        return self.tokenizer

    def get_device(self):
        """Get the device the model is on"""
        return self._device

    def cleanup(self):
        """Clean up resources and free memory"""
        if self.base_model is not None:
            del self.base_model
            self.base_model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model manager cleaned up")


class ModelCheckpoint:
    """Handles model checkpointing for crash recovery"""

    @staticmethod
    def save_checkpoint(model, optimizer, epoch: int, checkpoint_path: str,
                       additional_info: Optional[Dict] = None):
        """Save a training checkpoint

        Args:
            model: The model to save
            optimizer: The optimizer state to save
            epoch: Current epoch number
            checkpoint_path: Path to save checkpoint
            additional_info: Any additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """Load a training checkpoint

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
