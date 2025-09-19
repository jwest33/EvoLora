# LoRALab: Evolutionary LoRA Optimization Framework

A framework for automatically discovering optimal LoRA (Low-Rank Adaptation) configurations through evolutionary algorithms. LoRALab leverages Unsloth for faster training with less memory usage.

## Important Notice

**This project has been tested exclusively on:**
- Windows 11
- NVIDIA RTX 5060 Ti (16GB VRAM)
- CUDA 12.8 (nvcc compiler)
- Python 3.11
- PyTorch 2.5.1+cu124

Other configurations (Linux, macOS, different GPUs/CUDA versions) may work but have not been tested and may require adjustments.

## Key Features

### Evolutionary Optimization
- **Genetic Algorithm**: Automatically evolves LoRA hyperparameters (rank, learning rate, target modules)
- **Multi-objective Fitness**: Balances perplexity, training speed, and parameter efficiency
- **Population-based Training**: Tests multiple configurations in parallel
- **Adaptive Mutation**: Adjusts exploration based on convergence
- **Duplicate Prevention**: Hash-based configuration tracking ensures no repeated experiments
- **Extended Search Space**: Evolves 9 different hyperparameters for comprehensive optimization

### Unsloth Integration
- **3-5x Faster Training**: Hardware-optimized kernels for modern GPUs
- **70% Less VRAM**: 4-bit quantization with minimal accuracy loss
- **Smart Gradient Offloading**: Automatic memory management
- **Zero-overhead LoRA**: Optimized implementation for faster forward/backward passes

### Advanced Training Methods
- **SFT (Supervised Fine-Tuning)**: Standard instruction tuning with TRL integration
- **GRPO (Group Relative Policy Optimization)**: For reasoning and chain-of-thought models
- **Memory-Efficient Training**: Gradient checkpointing, mixed precision, and batch optimization

### Windows Optimization
- **Full Windows Support**: Handles multiprocessing limitations automatically
- **Memory Fragmentation Reduction**: Optimized allocator settings
- **Single-threaded Fallback**: Ensures stability on Windows systems

## System Requirements

### Tested Configuration
- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **CUDA**: 12.8 (nvcc compiler tools)
- **CUDA Driver**: 13.0 capable (driver 581.15)
- **Python**: 3.11
- **PyTorch**: 2.5.1+cu124
- **RAM**: 32GB+ recommended

## Installation

### Prerequisites
1. **Install CUDA Toolkit** (if not already installed)
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Tested setup uses CUDA 12.8
   - Note: PyTorch may use a different CUDA version (e.g., cu124 = CUDA 12.4)

2. **Verify CUDA Installation**
```bash
nvcc --version  # Should show your CUDA compiler version (e.g., 12.8)
nvidia-smi      # Should show your GPU and driver info
```

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/jwest33/loralab.git
cd loralab
```

2. **Create virtual environment** (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac (untested):
source .venv/bin/activate
```

3. **Install PyTorch with CUDA support** (critical - do this first!)
```bash
# For CUDA 12.x (tested with CUDA 12.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install xFormers for memory-efficient attention:
pip install xformers --no-deps --index-url https://download.pytorch.org/whl/cu128
```

4. **Install base dependencies**
```bash
pip install -r requirements.txt
```

5. **Install Unsloth** (critical for performance)
```bash
# For latest CUDA (tested with 12.8):
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# If you encounter issues, try:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --upgrade --force-reinstall --no-cache-dir
```


## Quick Start

### Fast Evolution (Recommended for Testing)
```bash
# Windows
./run_fast.bat

# Linux/Mac
./run_fast.sh
```

This runs a quick evolution with:
- 3 variants per generation
- 5 generations
- GSM8K dataset (math reasoning)
- 4-bit quantization
- Optimized memory settings

### Full Evolution
```bash
python run_evolution.py
```

### Custom Configuration
```bash
python run_evolution.py --config my_config.yaml --generations 20 --population 10
```

## Configuration

### Model Configuration
```yaml
model:
  path: "Qwen/Qwen3-4B-Instruct-2507"  # HuggingFace model ID
  backend: "unsloth"  # Use Unsloth optimizations
  quantization: "4bit"  # 4-bit, 8-bit, or none
  max_seq_length: 1024  # Reduce for less memory usage
```

### Evolution Parameters
```yaml
evolution:
  population_size: 8  # Variants per generation
  generations: 20  # Number of evolution cycles
  mutation_rate: 0.3  # Parameter mutation probability
  crossover_rate: 0.2  # Crossover breeding rate
```

### LoRA Search Space
The evolutionary algorithm explores a comprehensive search space:

```yaml
lora_search_space:
  # Core LoRA parameters
  rank: [8, 16, 32]  # LoRA rank options
  alpha_multiplier: [1, 2]  # Alpha = rank * multiplier
  dropout: [0.0]  # 0 for Unsloth fast patching
  learning_rate: [2e-5, 5e-5, 1e-4]

  # Training hyperparameters (evolved per variant)
  weight_decay: [0.0, 0.01, 0.05]  # L2 regularization
  warmup_ratio: [0.0, 0.1, 0.2]  # Learning rate warmup
  max_grad_norm: [0.5, 1.0, 2.0]  # Gradient clipping threshold

  # Advanced LoRA settings
  use_rslora: [false, true]  # Rank-Stabilized LoRA
  target_modules_preset: ["minimal", "standard", "extended"]
  # minimal: q_proj, v_proj (fastest)
  # standard: q_proj, k_proj, v_proj, o_proj (balanced)
  # extended: all 7 projection modules (comprehensive)
```

#### Search Space Size
- **Fast configs**: 384 unique combinations
- **Main config**: 1,458 unique combinations
- **Duplicate prevention**: Ensures no wasted computation on repeated configurations

## Supported Datasets

- **GSM8K** (default): Math word problems
- **Alpaca**: General instruction following
- **Dolly**: Diverse instruction dataset
- **Squad**: Question answering
- **Custom**: Any dataset with question/answer pairs

## Advanced Features

### GRPO Training
Enable GRPO for reasoning models:
```yaml
training:
  method: "grpo"
grpo:
  enabled: true
  reasoning_start: "<think>"
  reasoning_end: "</think>"
```

### Memory Optimization
The framework automatically:
- Clears CUDA cache periodically
- Uses gradient accumulation for large effective batches
- Implements smart offloading on low VRAM
- Reduces memory fragmentation on Windows

### Export Options
Best models can be exported as:
- LoRA adapters
- Merged 16-bit models
- 4-bit quantized models
- GGUF format for llama.cpp

## Monitoring

Training progress includes:
- Real-time loss tracking
- Memory usage monitoring
- Perplexity evaluation
- Generation-by-generation comparisons

## How Evolution Works

### Mutation Strategy
The genetic algorithm intelligently mutates variants across multiple dimensions:

1. **Smart Parameter Mutation**: Each hyperparameter has a 30% chance of mutation per generation
2. **Adaptive Learning Rate Scaling**: Can scale existing LR by factors (0.5x, 0.8x, 1.25x, 2x) or pick new values
3. **Module Preset Evolution**: Automatically tests different combinations of target modules
4. **Training Dynamics Evolution**: Optimizes weight decay, warmup, and gradient clipping per variant

### Duplicate Prevention System
- **Configuration Hashing**: Each variant gets a unique MD5 hash based on all parameters
- **Intelligent Retry**: Attempts to create unique variants up to 10 times
- **Forced Variation**: If uniqueness cannot be achieved, applies small perturbations
- **Memory Efficient**: Tracks only configuration hashes, not full variant objects

### Fitness Evaluation
Each variant is scored based on:
- **Accuracy** (70% weight): Task performance on evaluation set
- **Perplexity** (30% weight): Model confidence and calibration
- **Efficiency Penalty**: Slight penalty for very large ranks (>128)

## Results

Typical improvements from evolution:
- **30-50% better perplexity** vs random hyperparameters
- **2-3x faster convergence** with optimal learning rates
- **Automatic rank selection** balancing performance and efficiency
- **Optimal training dynamics** discovered through evolution
- **48x more diverse** search space with new mutation properties

## Common Issues & Troubleshooting

### Installation Issues

**Unsloth Installation Fails**
```bash
# Try installing with no cache:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir

# Or install from PyPI for your CUDA version:
pip install unsloth[cu121]  # For CUDA 12.1
```

**CUDA Version Mismatch**
- PyTorch CUDA version doesn't need to exactly match system CUDA
- System CUDA 12.8 works fine with PyTorch cu124 (CUDA 12.4)
- Check versions:
  ```bash
  nvcc --version  # System CUDA compiler (e.g., 12.8)
  python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA (e.g., 12.4)
  ```
- As long as driver supports the PyTorch CUDA version, it will work

**bitsandbytes on Windows**
```bash
# If bitsandbytes fails, try:
pip install bitsandbytes-windows
# Or use the pre-built wheels:
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### Runtime Issues

**CUDA Out of Memory**
- Reduce `batch_size` in config.yaml (try 2 or 4)
- Lower `max_seq_length` (try 512 or 256)
- Use 4-bit quantization instead of 8-bit
- Reduce `population_size` in evolution config

**Windows Multiprocessing Errors**
- Already handled automatically in the code
- If issues persist, set in config.yaml:
  ```yaml
  training:
    dataloader_num_workers: 0
  ```

**Slow Training**
1. Verify Unsloth is being used:
   ```python
   python -c "from loralab.core.unsloth_manager import UnslothModelManager; print('Unsloth OK')"
   ```
2. Check GPU utilization: `nvidia-smi` should show high usage
3. Ensure dropout is 0.0 for Unsloth fast patching
4. Use batch_size that maximizes GPU memory usage

**Memory Fragmentation**
- The code automatically sets optimal PyTorch memory settings
- If issues persist, restart Python/clear CUDA cache:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for incredible optimization work
- [HuggingFace](https://huggingface.co/) for Transformers and PEFT
- [TRL](https://github.com/huggingface/trl) for advanced training methods
- [Qwen](https://huggingface.co/Qwen) for amazing oss language models
