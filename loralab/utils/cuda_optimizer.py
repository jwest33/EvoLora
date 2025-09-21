"""CUDA Optimization for R-Zero Training

Optimizes CUDA settings for RTX 5060 Ti 16GB and similar GPUs
to maximize training efficiency and prevent OOM errors.
"""

import logging
import os
import torch
import gc
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CUDAOptimizer:
    """CUDA optimization utilities for R-Zero training"""

    def __init__(self, config):
        """Initialize CUDA optimizer

        Args:
            config: CUDAConfig with optimization settings
        """
        self.config = config
        self.device = None
        self.vram_gb = 16  # Default for RTX 5060 Ti

        # Detect and setup CUDA
        self._setup_cuda()

    def _setup_cuda(self):
        """Setup CUDA device and settings"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = torch.device("cpu")
            return

        # Get device properties
        self.device = torch.device("cuda")
        device_props = torch.cuda.get_device_properties(0)

        # Calculate available VRAM
        total_vram_bytes = device_props.total_memory
        self.vram_gb = total_vram_bytes / (1024 ** 3)

        logger.info(f"CUDA Device: {device_props.name}")
        logger.info(f"VRAM: {self.vram_gb:.1f} GB")
        logger.info(f"Compute Capability: {device_props.major}.{device_props.minor}")

        # Set CUDA settings
        self._configure_cuda_settings()

    def _configure_cuda_settings(self):
        """Configure optimal CUDA settings"""
        # Enable TF32 for better performance on Ampere+ GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for Ampere+ GPU")

        # Enable cudnn benchmarking for consistent workloads
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set memory fraction
        if self.config.max_memory_reserved < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.config.max_memory_reserved)
            logger.info(f"CUDA memory fraction set to {self.config.max_memory_reserved}")

        # Disable CUDA graphs for GRPO (can cause issues)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    def optimize_model_for_training(self, model, model_type: str = "solver"):
        """Apply optimizations to a model for training

        Args:
            model: The model to optimize
            model_type: "challenger" or "solver"

        Returns:
            Optimized model
        """
        if not torch.cuda.is_available():
            return model

        # Move model to CUDA
        model = model.to(self.device)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info(f"Gradient checkpointing enabled for {model_type}")
            elif hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
                logger.info(f"Input gradients enabled for {model_type}")

        # Set mixed precision if available
        if self.config.mixed_precision == "bf16":
            # BF16 is preferred for stability
            if torch.cuda.is_bf16_supported():
                model = model.to(torch.bfloat16)
                logger.info(f"BF16 mixed precision enabled for {model_type}")
        elif self.config.mixed_precision == "fp16":
            model = model.half()
            logger.info(f"FP16 mixed precision enabled for {model_type}")

        # Apply torch.compile if enabled (disabled by default for GRPO)
        if self.config.use_torch_compile:
            try:
                model = torch.compile(model, mode=self.config.compile_mode)
                logger.info(f"Torch compile applied to {model_type} with mode {self.config.compile_mode}")
            except Exception as e:
                logger.warning(f"Failed to compile {model_type}: {e}")

        return model

    def get_optimal_batch_size(self, model_type: str = "solver") -> int:
        """Get optimal batch size for current VRAM

        Args:
            model_type: "challenger" or "solver"

        Returns:
            Recommended batch size
        """
        if model_type == "challenger":
            base_batch_size = self.config.max_batch_size_challenger
        else:
            base_batch_size = self.config.max_batch_size_solver

        # Adjust based on available VRAM
        if self.vram_gb < 8:
            return max(1, base_batch_size // 4)
        elif self.vram_gb < 12:
            return max(1, base_batch_size // 2)
        elif self.vram_gb < 24:
            return base_batch_size
        else:
            return base_batch_size * 2

    def clear_cache(self, force: bool = False):
        """Clear CUDA cache

        Args:
            force: Force garbage collection
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            if force:
                gc.collect()
                torch.cuda.empty_cache()

            # Log memory stats
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.debug(f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def monitor_memory(self) -> Dict[str, float]:
        """Monitor CUDA memory usage

        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {}

        stats = {
            "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024 ** 3),
        }

        # Check if we're close to OOM
        if stats["free_gb"] < 1.0:
            logger.warning(f"Low CUDA memory: {stats['free_gb']:.2f}GB free")
            self.clear_cache(force=True)

        return stats

    def setup_training_environment(self):
        """Setup environment variables for optimal training"""
        # Disable debugging features for performance
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

        # Enable NCCL optimizations (for multi-GPU, but doesn't hurt single GPU)
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

        # Disable Python garbage collector during training (re-enable manually)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

        # Unsloth specific settings
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

        # Disable torch compile for GRPO stability
        if not self.config.use_torch_compile:
            os.environ['TORCHDYNAMO_DISABLE'] = '1'

        logger.info("Training environment configured")

    def get_memory_config_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get memory configuration based on model size

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with memory settings
        """
        config = {
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "mixed_precision": self.config.mixed_precision,
        }

        # Adjust based on model size
        if "3b" in model_name.lower() or "4b" in model_name.lower():
            # 3-4B parameter models
            config["max_seq_length"] = 2048
            config["batch_size"] = 4 if self.vram_gb >= 16 else 2
            config["gradient_accumulation_steps"] = 32

        elif "270m" in model_name.lower() or "2b" in model_name.lower():
            # Smaller models like Gemma-270m
            config["max_seq_length"] = 2048
            config["batch_size"] = 8 if self.vram_gb >= 16 else 4
            config["gradient_accumulation_steps"] = 16

        else:
            # Default conservative settings
            config["max_seq_length"] = 1024
            config["batch_size"] = 2
            config["gradient_accumulation_steps"] = 64

        logger.info(f"Memory config for {model_name}: {config}")
        return config

    def handle_oom_error(self, error: Exception, model_type: str = "solver"):
        """Handle OOM errors gracefully

        Args:
            error: The OOM exception
            model_type: Type of model that caused OOM
        """
        logger.error(f"OOM error in {model_type}: {error}")

        # Clear cache
        self.clear_cache(force=True)

        # Suggest adjustments
        suggestions = [
            f"Reduce batch size for {model_type}",
            "Enable gradient checkpointing",
            "Reduce max sequence length",
            "Use stronger quantization (4-bit)",
            "Increase gradient accumulation steps",
        ]

        logger.info("Suggestions to avoid OOM:")
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"  {i}. {suggestion}")

        # Return reduced settings
        return {
            "batch_size": 1,
            "gradient_accumulation_steps": 128,
            "max_seq_length": 512,
        }

    def optimize_dataloader(self, dataloader, model_type: str = "solver"):
        """Optimize dataloader settings

        Args:
            dataloader: The dataloader to optimize
            model_type: Type of model using this dataloader

        Returns:
            Optimized dataloader
        """
        # Set num_workers based on system
        if os.name == 'nt':  # Windows
            num_workers = 0  # Avoid multiprocessing issues on Windows
        else:
            num_workers = min(4, os.cpu_count() // 2)

        dataloader.num_workers = num_workers
        dataloader.pin_memory = torch.cuda.is_available()
        dataloader.persistent_workers = num_workers > 0

        logger.info(f"Dataloader optimized for {model_type}: workers={num_workers}, pin_memory={dataloader.pin_memory}")

        return dataloader


def create_cuda_optimizer(config) -> CUDAOptimizer:
    """Factory function to create CUDA optimizer

    Args:
        config: CUDAConfig with settings

    Returns:
        Configured CUDAOptimizer instance
    """
    optimizer = CUDAOptimizer(config)
    optimizer.setup_training_environment()
    return optimizer