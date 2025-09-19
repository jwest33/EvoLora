"""Memory monitoring utilities for tracking GPU and RAM usage during training"""

import torch
import psutil
import GPUtil
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor GPU and system memory usage"""

    def __init__(self, log_interval: int = 10):
        """Initialize memory monitor

        Args:
            log_interval: How often to log memory stats (in batches)
        """
        self.log_interval = log_interval
        self.batch_count = 0
        self.start_time = time.time()

        # Track peak usage
        self.peak_gpu_memory = 0
        self.peak_ram_usage = 0

        # Get initial memory state
        self.initial_ram = self._get_ram_usage()
        self.initial_gpu = self._get_gpu_memory() if torch.cuda.is_available() else 0

    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0}

        # PyTorch's view of GPU memory
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB

        # System's view of GPU memory
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming single GPU
                total_memory = gpu.memoryTotal / 1024  # GB
                used_memory = gpu.memoryUsed / 1024    # GB
                free_memory = gpu.memoryFree / 1024    # GB
                utilization = gpu.memoryUtil * 100     # Percentage
            else:
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                used_memory = reserved
                free_memory = total_memory - used_memory
                utilization = (used_memory / total_memory) * 100
        except:
            # Fallback if GPUtil fails
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            used_memory = reserved
            free_memory = total_memory - used_memory
            utilization = (used_memory / total_memory) * 100

        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total_memory,
            'used': used_memory,
            'free': free_memory,
            'utilization': utilization
        }

    def _get_ram_usage(self) -> Dict[str, float]:
        """Get current RAM usage"""
        ram = psutil.virtual_memory()
        process = psutil.Process()

        return {
            'total': ram.total / 1024**3,         # Total RAM in GB
            'used': ram.used / 1024**3,           # System-wide used RAM
            'available': ram.available / 1024**3,  # Available RAM
            'percent': ram.percent,                # Usage percentage
            'process': process.memory_info().rss / 1024**3  # This process's RAM usage
        }

    def check_memory_pressure(self) -> Dict[str, any]:
        """Check for signs of memory pressure and offloading"""
        gpu_mem = self._get_gpu_memory()
        ram = self._get_ram_usage()

        # Update peaks
        self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem['used'])
        self.peak_ram_usage = max(self.peak_ram_usage, ram['process'])

        # Detect potential offloading
        signs_of_offloading = []

        # Check if GPU memory is nearly full
        if gpu_mem['utilization'] > 90:
            signs_of_offloading.append("GPU memory >90% utilized")

        # Check if process RAM usage increased significantly since start
        ram_increase = ram['process'] - self.initial_ram['process']
        if ram_increase > 2.0:  # More than 2GB increase
            signs_of_offloading.append(f"Process RAM increased by {ram_increase:.2f}GB")

        # Check if there's a mismatch between allocated and reserved GPU memory
        if gpu_mem['reserved'] - gpu_mem['allocated'] > 1.0:  # More than 1GB difference
            signs_of_offloading.append(f"GPU memory fragmentation: {gpu_mem['reserved'] - gpu_mem['allocated']:.2f}GB gap")

        # Check system RAM usage
        if ram['percent'] > 80:
            signs_of_offloading.append(f"System RAM usage high: {ram['percent']:.1f}%")

        return {
            'gpu_memory': gpu_mem,
            'ram': ram,
            'ram_increase_gb': ram_increase,
            'signs_of_offloading': signs_of_offloading,
            'likely_offloading': len(signs_of_offloading) > 0,
            'peak_gpu_gb': self.peak_gpu_memory,
            'peak_ram_gb': self.peak_ram_usage
        }

    def log_memory_status(self, batch_idx: Optional[int] = None, variant_id: str = ""):
        """Log current memory status"""
        self.batch_count += 1

        # Only log at intervals
        if self.batch_count % self.log_interval != 0:
            return

        status = self.check_memory_pressure()

        # Format log message
        gpu = status['gpu_memory']
        ram = status['ram']

        log_msg = (f"[Memory] {variant_id} "
                  f"GPU: {gpu['allocated']:.2f}/{gpu['reserved']:.2f}/{gpu['total']:.2f}GB "
                  f"({gpu['utilization']:.1f}%) | "
                  f"RAM: {ram['process']:.2f}GB (System: {ram['percent']:.1f}%)")

        if status['likely_offloading']:
            log_msg += f" | WARNING: {', '.join(status['signs_of_offloading'])}"
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Clear PyTorch's cache periodically to prevent fragmentation
        if gpu['utilization'] > 85:
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache due to high memory usage")

    def get_summary(self) -> str:
        """Get a summary of memory usage"""
        status = self.check_memory_pressure()
        elapsed = time.time() - self.start_time

        summary = [
            "\n" + "="*60,
            "MEMORY USAGE SUMMARY",
            "="*60,
            f"Duration: {elapsed/60:.1f} minutes",
            f"\nGPU Memory:",
            f"  Current: {status['gpu_memory']['used']:.2f}/{status['gpu_memory']['total']:.2f}GB ({status['gpu_memory']['utilization']:.1f}%)",
            f"  Peak: {self.peak_gpu_memory:.2f}GB",
            f"  Allocated: {status['gpu_memory']['allocated']:.2f}GB",
            f"  Reserved: {status['gpu_memory']['reserved']:.2f}GB",
            f"\nSystem RAM:",
            f"  Process: {status['ram']['process']:.2f}GB (Peak: {self.peak_ram_usage:.2f}GB)",
            f"  System: {status['ram']['used']:.2f}/{status['ram']['total']:.2f}GB ({status['ram']['percent']:.1f}%)",
            f"  Increase since start: {status['ram_increase_gb']:.2f}GB"
        ]

        if status['likely_offloading']:
            summary.append(f"\n⚠️  Possible Memory Offloading Detected:")
            for sign in status['signs_of_offloading']:
                summary.append(f"  - {sign}")
        else:
            summary.append("\n✓ No signs of memory offloading detected")

        summary.append("="*60 + "\n")

        return "\n".join(summary)


def diagnose_memory_usage(model, sample_batch_size: int = 16):
    """Diagnose memory usage for a model

    Args:
        model: The model to diagnose
        sample_batch_size: Batch size to test with
    """
    if not torch.cuda.is_available():
        print("CUDA not available, cannot diagnose GPU memory")
        return

    device = next(model.parameters()).device

    print("\n" + "="*60)
    print("MEMORY DIAGNOSTIC")
    print("="*60)

    # Clear cache first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Model memory
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    trainable_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024**3

    print(f"\nModel Memory:")
    print(f"  Total parameters memory: {model_memory:.3f}GB")
    print(f"  Trainable parameters memory: {trainable_memory:.3f}GB")

    # Current GPU state
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3

    print(f"\nCurrent GPU Memory:")
    print(f"  Allocated: {allocated:.3f}GB")
    print(f"  Reserved: {reserved:.3f}GB")

    # Estimate batch memory
    # Assuming sequence length of 256 and vocab size of ~150k for Qwen
    seq_length = 256
    vocab_size = 151936  # Qwen's vocab size

    # Forward pass memory (rough estimate)
    # Input embeddings + attention matrices + output logits
    input_memory = (sample_batch_size * seq_length * 4096 * 2) / 1024**3  # embeddings
    attention_memory = (sample_batch_size * 32 * seq_length * seq_length * 2) / 1024**3  # attention heads
    output_memory = (sample_batch_size * seq_length * vocab_size * 2) / 1024**3  # logits

    estimated_batch_memory = input_memory + attention_memory + output_memory

    print(f"\nEstimated Memory per Batch (size={sample_batch_size}):")
    print(f"  Input embeddings: {input_memory:.3f}GB")
    print(f"  Attention: {attention_memory:.3f}GB")
    print(f"  Output logits: {output_memory:.3f}GB")
    print(f"  Total estimate: {estimated_batch_memory:.3f}GB")

    # GPU capacity
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    available = total_memory - reserved

    print(f"\nGPU Capacity:")
    print(f"  Total: {total_memory:.3f}GB")
    print(f"  Available: {available:.3f}GB")
    print(f"  Reserved for this process: {reserved:.3f}GB")

    # Recommendations
    print(f"\nRecommendations:")
    if available < estimated_batch_memory:
        print(f"  ⚠️  Batch size {sample_batch_size} may cause OOM!")
        safe_batch = int(available / (estimated_batch_memory / sample_batch_size))
        print(f"  💡 Try batch size <= {safe_batch}")
    else:
        print(f"  ✓ Batch size {sample_batch_size} should fit in memory")
        max_batch = int(available / (estimated_batch_memory / sample_batch_size))
        print(f"  💡 Maximum batch size estimate: {max_batch}")

    print("="*60 + "\n")