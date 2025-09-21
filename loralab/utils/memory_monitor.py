"""Memory monitoring utilities for tracking GPU and RAM usage during training"""

import torch
import psutil
import GPUtil
import logging
from typing import Dict, Optional
import time
from .cli_formatter import CLIFormatter

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

        # Use CLIFormatter for colored output
        gpu = status['gpu_memory']
        ram = status['ram']

        # Print formatted memory status
        CLIFormatter.print_memory_status(gpu, ram, variant_id)

        # Print warnings if offloading detected
        if status['likely_offloading']:
            CLIFormatter.print_warning(f"Memory pressure detected: {', '.join(status['signs_of_offloading'])}")

        # Clear PyTorch's cache periodically to prevent fragmentation
        if gpu['utilization'] > 85:
            torch.cuda.empty_cache()
            CLIFormatter.print_info("Cleared CUDA cache due to high memory usage")


    def get_summary(self) -> str:
        """Get a summary of memory usage"""
        status = self.check_memory_pressure()
        elapsed = time.time() - self.start_time
        gpu_mem = status['gpu_memory']
        ram = status['ram']

        # Build formatted summary
        from io import StringIO
        import sys

        # Capture formatted output
        old_stdout = sys.stdout
        sys.stdout = output_buffer = StringIO()

        # Print formatted summary
        CLIFormatter.print_header("MEMORY USAGE SUMMARY")
        CLIFormatter.print_info(f"Duration: {CLIFormatter.format_time(elapsed)}")

        # GPU Memory section
        CLIFormatter.print_subheader("GPU Memory")
        utilization_percent = gpu_mem['utilization']
        CLIFormatter.print_metric("Current", gpu_mem['used'], f"/{gpu_mem['total']:.2f}GB ({utilization_percent:.1f}%)",
                                 good_threshold=70, bad_threshold=85)
        CLIFormatter.print_metric("Peak", self.peak_gpu_memory, "GB")
        CLIFormatter.print_metric("Allocated", gpu_mem['allocated'], "GB")
        CLIFormatter.print_metric("Reserved", gpu_mem['reserved'], "GB")

        # System RAM section
        CLIFormatter.print_subheader("System RAM")
        CLIFormatter.print_metric("Process", ram['process'], f"GB (Peak: {self.peak_ram_usage:.2f}GB)")
        CLIFormatter.print_metric("System", ram['used'], f"/{ram['total']:.2f}GB ({ram['percent']:.1f}%)",
                                 good_threshold=70, bad_threshold=85)
        CLIFormatter.print_metric("Increase since start", status['ram_increase_gb'], "GB",
                                 good_threshold=2.0, bad_threshold=4.0)

        # Offloading detection
        if status['likely_offloading']:
            CLIFormatter.print_warning("Possible Memory Offloading Detected:")
            for sign in status['signs_of_offloading']:
                CLIFormatter.print_list_item(sign, level=1)
        else:
            CLIFormatter.print_success("No signs of memory offloading detected")

        # Get the formatted output
        sys.stdout = old_stdout
        return output_buffer.getvalue()


def diagnose_memory_usage(model, sample_batch_size: int = 16):
    """Diagnose memory usage for a model

    Args:
        model: The model to diagnose
        sample_batch_size: Batch size to test with
    """
    if not torch.cuda.is_available():
        CLIFormatter.print_error("CUDA not available, cannot diagnose GPU memory")
        return

    device = next(model.parameters()).device

    CLIFormatter.print_header("MEMORY DIAGNOSTIC")

    # Clear cache first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Model memory
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    trainable_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024**3

    CLIFormatter.print_subheader("Model Memory")
    CLIFormatter.print_metric("Total parameters memory", model_memory, "GB")
    CLIFormatter.print_metric("Trainable parameters memory", trainable_memory, "GB")

    # Current GPU state
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3

    CLIFormatter.print_subheader("Current GPU Memory")
    CLIFormatter.print_metric("Allocated", allocated, "GB", good_threshold=0.7*torch.cuda.get_device_properties(device).total_memory/1024**3, bad_threshold=0.9*torch.cuda.get_device_properties(device).total_memory/1024**3)
    CLIFormatter.print_metric("Reserved", reserved, "GB", good_threshold=0.7*torch.cuda.get_device_properties(device).total_memory/1024**3, bad_threshold=0.9*torch.cuda.get_device_properties(device).total_memory/1024**3)

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

    CLIFormatter.print_subheader(f"Estimated Memory per Batch (size={sample_batch_size})")
    CLIFormatter.print_metric("Input embeddings", input_memory, "GB")
    CLIFormatter.print_metric("Attention", attention_memory, "GB")
    CLIFormatter.print_metric("Output logits", output_memory, "GB")
    CLIFormatter.print_metric("Total estimate", estimated_batch_memory, "GB")

    # GPU capacity
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    available = total_memory - reserved

    CLIFormatter.print_subheader("GPU Capacity")
    CLIFormatter.print_metric("Total", total_memory, "GB")
    CLIFormatter.print_metric("Available", available, "GB", good_threshold=0.3*total_memory, bad_threshold=0.1*total_memory)
    CLIFormatter.print_metric("Reserved for this process", reserved, "GB")

    # Recommendations
    CLIFormatter.print_subheader("Recommendations")
    if available < estimated_batch_memory:
        CLIFormatter.print_warning(f"Batch size {sample_batch_size} may cause OOM!")
        safe_batch = int(available / (estimated_batch_memory / sample_batch_size))
        CLIFormatter.print_info(f"Try batch size <= {safe_batch}")
    else:
        CLIFormatter.print_success(f"Batch size {sample_batch_size} should fit in memory")
        max_batch = int(available / (estimated_batch_memory / sample_batch_size))
        CLIFormatter.print_info(f"Maximum batch size estimate: {max_batch}")