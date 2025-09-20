"""Unsloth Model Manager for optimized LoRA training

Provides FastLanguageModel integration with 4-bit/8-bit quantization,
optimized training, and advanced LoRA features from Unsloth.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os

# Use centralized Unsloth initialization to handle Windows compatibility
from ..utils.unsloth_config import init_unsloth, is_unsloth_available

# Initialize Unsloth before any other imports
UNSLOTH_AVAILABLE = init_unsloth()

# Import Unsloth components if available
if UNSLOTH_AVAILABLE:
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
    except ImportError:
        UNSLOTH_AVAILABLE = False
        FastLanguageModel = None
        get_chat_template = None
        train_on_responses_only = None
else:
    FastLanguageModel = None
    get_chat_template = None
    train_on_responses_only = None

import torch  # Import torch AFTER unsloth

try:
    from ..utils.cli_formatter import CLIFormatter
    USE_FORMATTER = True
except ImportError:
    USE_FORMATTER = False

logger = logging.getLogger(__name__)


class UnslothModelManager:
    """Manages Unsloth FastLanguageModel for optimized LoRA training"""

    # Supported 4-bit models from Unsloth
    FOURBIT_MODELS = [
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/Phi-4",
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
    ]

    def __init__(self, config: Dict[str, Any]):
        """Initialize with model configuration

        Args:
            config: Model configuration dictionary containing:
                - path: HuggingFace model ID or local path
                - quantization: "4bit", "8bit", or "none"
                - max_seq_length: Maximum sequence length
                - use_gradient_checkpointing: Enable gradient checkpointing
                - chat_template: Chat template to use (optional)
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth is required for UnslothModelManager. Install with: pip install unsloth")

        self.config = config
        self.model = None
        self.tokenizer = None
        self._device = None

    def load_base_model(self) -> Tuple[Any, Any]:
        """Load the base model and tokenizer using Unsloth

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.config['path']
        quantization = self.config.get('quantization', '4bit')
        max_seq_length = self.config.get('max_seq_length', 2048)

        logger.info(f"Loading Unsloth model: {model_path} with {quantization} quantization")

        try:
            # Load model with Unsloth optimizations
            load_in_4bit = quantization == '4bit'
            load_in_8bit = quantization == '8bit'

            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Determine dtype from config or model type
            dtype = None
            config_dtype = self.config.get('torch_dtype')

            if config_dtype:
                # Use explicit dtype from config
                if config_dtype == 'float16':
                    dtype = torch.float16
                elif config_dtype == 'bfloat16':
                    dtype = torch.bfloat16
                elif config_dtype == 'float32':
                    dtype = torch.float32
                logger.info(f"Using configured dtype: {config_dtype}")
            elif 'gemma' in model_path.lower() and not (load_in_4bit or load_in_8bit):
                # Default to float16 for Gemma to avoid dtype conflicts
                dtype = torch.float16
                logger.info("Using float16 for Gemma model (avoiding BFloat16 conflicts)")

            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                dtype=dtype,  # Use explicit dtype
                # No vLLM settings since we're not using it
            )

            # Disable gradient checkpointing for base model to reduce memory fragmentation
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()

            # Apply chat template if specified
            chat_template = self.config.get('chat_template')
            if chat_template:
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template=chat_template
                )
                logger.info(f"Applied chat template: {chat_template}")

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Get device
            self._device = next(self.model.parameters()).device

            if USE_FORMATTER:
                CLIFormatter.print_success(f"Unsloth model loaded successfully on {self._device}")
            else:
                logger.info(f"Unsloth model loaded successfully on {self._device}")

            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            if USE_FORMATTER:
                CLIFormatter.print_info(f"Total parameters: {total_params:,}")
                CLIFormatter.print_info(f"Quantization: {quantization}")
                CLIFormatter.print_info(f"Max sequence length: {max_seq_length}")
            else:
                logger.info(f"Total parameters: {total_params:,}")

            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load Unsloth model: {e}")
            raise

    def create_lora_variant(self, lora_config: Dict[str, Any]) -> Any:
        """Create a new LoRA variant with Unsloth optimizations

        Args:
            lora_config: LoRA configuration containing:
                - rank: LoRA rank (r)
                - alpha_multiplier: Alpha = rank * multiplier
                - dropout: Dropout rate
                - target_modules: List of modules to apply LoRA to
                - use_rslora: Use Rank-Stabilized LoRA
                - use_gradient_checkpointing: Enable gradient checkpointing

        Returns:
            PEFT model with LoRA applied
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        # Calculate alpha based on multiplier
        rank = lora_config['rank']
        alpha_multiplier = lora_config.get('alpha_multiplier', 2)
        lora_alpha = rank * alpha_multiplier

        # Default target modules for Unsloth
        target_modules = lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])

        # Gradient checkpointing mode
        gradient_checkpointing = lora_config.get('use_gradient_checkpointing', True)
        if gradient_checkpointing:
            # Use "unsloth" for 30% less VRAM and 2x larger batch sizes
            gradient_checkpointing = "unsloth"

        # Apply LoRA with Unsloth optimizations
        lora_model = FastLanguageModel.get_peft_model(
            self.model,
            r=rank,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_config.get('dropout', 0),
            bias="none",
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=3407,
            use_rslora=lora_config.get('use_rslora', False),
            loftq_config=lora_config.get('loftq_config', None),
        )

        # Ensure model is in training mode
        lora_model.train()

        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in lora_model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        logger.info(f"LoRA variant created with Unsloth optimizations:")
        logger.info(f"  Rank: {rank}, Alpha: {lora_alpha} (multiplier: {alpha_multiplier})")
        logger.info(f"  Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"  Gradient checkpointing: {gradient_checkpointing}")
        logger.info(f"  RSLoRA: {lora_config.get('use_rslora', False)}")

        return lora_model

    def save_model(self, model, save_path: str, save_method: str = "lora"):
        """Save the model using Unsloth's optimized saving methods

        Args:
            model: The model to save
            save_path: Path to save the model
            save_method: "lora", "merged_16bit", "merged_4bit", or "gguf"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_method == "lora":
            model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            logger.info(f"Saved LoRA adapters to {save_path}")

        elif save_method == "merged_16bit":
            model.save_pretrained_merged(
                str(save_path),
                self.tokenizer,
                save_method="merged_16bit"
            )
            logger.info(f"Saved merged 16-bit model to {save_path}")

        elif save_method == "merged_4bit":
            model.save_pretrained_merged(
                str(save_path),
                self.tokenizer,
                save_method="merged_4bit"
            )
            logger.info(f"Saved merged 4-bit model to {save_path}")

        elif save_method.startswith("gguf"):
            # Extract quantization method if specified
            quant_method = "q8_0"  # Default
            if "_" in save_method:
                quant_method = save_method.split("_", 1)[1]

            model.save_pretrained_gguf(
                str(save_path),
                self.tokenizer,
                quantization_method=quant_method
            )
            logger.info(f"Saved GGUF model ({quant_method}) to {save_path}")

        else:
            raise ValueError(f"Unknown save method: {save_method}")

    def export_to_ollama(self, model, model_name: str, quantization: str = "q4_k_m"):
        """Export model for use with Ollama

        Args:
            model: The model to export
            model_name: Name for the Ollama model
            quantization: GGUF quantization method
        """
        # Save as GGUF first
        gguf_path = f"models/ollama/{model_name}"
        self.save_model(model, gguf_path, f"gguf_{quantization}")

        logger.info(f"Model exported for Ollama: {gguf_path}")
        logger.info(f"To use with Ollama: ollama create {model_name} -f {gguf_path}/Modelfile")

    def get_tokenizer(self):
        """Get the tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_base_model() first.")
        return self.tokenizer

    def get_model(self):
        """Get the base model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        return self.model

    def get_device(self):
        """Get the device the model is on"""
        return self._device

    def cleanup(self):
        """Clean up resources and free memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Unsloth model manager cleaned up")


def create_model_manager(config: Dict[str, Any]) -> Any:
    """Factory function to create appropriate model manager

    Args:
        config: Model configuration

    Returns:
        Either UnslothModelManager or regular ModelManager based on config
    """
    backend = config.get('backend', 'transformers')

    if backend == 'unsloth' and UNSLOTH_AVAILABLE:
        return UnslothModelManager(config)
    else:
        # Fall back to regular ModelManager
        from .model_manager import ModelManager
        return ModelManager(config)
