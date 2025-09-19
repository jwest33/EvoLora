"""Memory optimization utilities for Unsloth training

Provides functions to optimize memory usage and reduce fragmentation.
"""

import torch
import gc
import os
import logging
from .cli_formatter import CLIFormatter

logger = logging.getLogger(__name__)

def optimize_memory():
    """Optimize memory before training starts"""

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Set memory fraction to prevent over-allocation
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Log current memory state with formatted output
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        utilization = (reserved / total) * 100 if total > 0 else 0

        # Use formatted memory status display
        gpu_memory = {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'utilization': utilization
        }

        CLIFormatter.print_info("GPU Memory optimized")
        CLIFormatter.print_metric("Allocated", allocated, "GB", good_threshold=0.7*total, bad_threshold=0.9*total)
        CLIFormatter.print_metric("Reserved", reserved, "GB", good_threshold=0.7*total, bad_threshold=0.9*total)
        CLIFormatter.print_metric("Total", total, "GB")

    # Set environment variables for better memory management
    # Smaller split size reduces fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async execution

def set_memory_efficient_settings():
    """Configure PyTorch for memory-efficient training"""

    # Enable TF32 for better performance (RTX 30xx and newer)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Use deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune for better performance

    # Set autocast settings for mixed precision
    torch.set_float32_matmul_precision('high')

def cleanup_after_variant():
    """Clean up memory after training a variant"""

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache multiple times
    if torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Force Python garbage collection
    for _ in range(2):
        gc.collect()

    logger.debug("Memory cleanup completed")