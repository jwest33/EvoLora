# EvoLoRA: Evolutionary LoRA Optimization Framework

A framework for automatically discovering optimal LoRA (Low-Rank Adaptation) configurations through evolutionary algorithms. EvoLoRA leverages Unsloth for faster training with less memory usage.

## Key Features

### Evolutionary Optimization
- **Genetic Algorithm**: Automatically evolves LoRA hyperparameters (rank, learning rate, target modules)
- **Multi-objective Fitness**: Balances accuracy and perplexity metrics
- **Population-based Training**: Tests multiple configurations in parallel
- **Adaptive Mutation**: Adjusts exploration based on convergence
- **Duplicate Prevention**: Hash-based configuration tracking ensures no repeated experiments
- **Extended Search Space**: Evolves 9 different hyperparameters for comprehensive optimization

### Unsloth Integration
- **3-5x Faster Training**: Hardware-optimized kernels for modern GPUs
- **70% Less VRAM**: 4-bit quantization support with minimal accuracy loss
- **Smart Gradient Offloading**: Automatic memory management
- **Zero-overhead LoRA**: Optimized implementation for faster forward/backward passes

### Advanced Training Methods
- **SFT (Supervised Fine-Tuning)**: Standard instruction tuning with TRL integration
- **GRPO (Group Relative Policy Optimization)**: For reasoning and chain-of-thought models
- **Pre-training on Format**: Optional format pre-training for GRPO to improve compliance
- **Memory-Efficient Training**: Gradient checkpointing, mixed precision, and batch optimization

### Windows Optimization
- **Full Windows Support**: Handles multiprocessing limitations automatically
- **Memory Fragmentation Reduction**: Optimized allocator settings
- **Unicode-safe Logging**: ASCII-only output for Windows console compatibility

## Installation

### Prerequisites
1. **Python 3.11+**
2. **CUDA-capable GPU** with at least 8GB VRAM
3. **CUDA Toolkit** (only tested on 12.8)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/EvoLoRA.git
cd EvoLoRA
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

3. **Install PyTorch with CUDA support**
```bash
# For CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Install Unsloth**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Quick Start

### Run Evolution with Default Settings
```bash
# Windows - Run Gemma3 model with GRPO
run_gemma.bat grpo

### Command Line Interface
```bash
# Basic evolution
python -m EvoLoRA.cli_evolution evolve --config EvoLoRA/config/qwen_config.yaml

# With GRPO training
python -m EvoLoRA.cli_evolution evolve --config EvoLoRA/config/gemma_config.yaml --grpo

# Custom parameters
python -m EvoLoRA.cli_evolution evolve \
    --config EvoLoRA/config/qwen_config.yaml \
    --generations 20 \
    --population 10 \
    --output my_experiment
```

## Configuration

### Model Configuration
Models are configured via YAML files in `EvoLoRA/config/`:

```yaml
# For small models (e.g., Gemma3-270M)
model:
  path: "unsloth/gemma-3-270m-it"  # HuggingFace model ID
  backend: "unsloth"                # Use Unsloth optimizations
  quantization: "none"              # No quantization for small models
  max_seq_length: 2048              # Maximum sequence length
  chat_template: "gemma3"           # Model-specific chat template

# For larger models (e.g., Qwen3-4B)
model:
  path: "Qwen/Qwen3-4B-Instruct-2507"
  quantization: "4bit"              # 4-bit for memory efficiency
  load_in_4bit: true                # Enable 4-bit loading
```

### Evolution Parameters
```yaml
evolution:
  population_size: 12    # More variants for better exploration
  generations: 15        # Sufficient generations for convergence
  keep_top: 3           # Keep more top performers
  mutation_rate: 0.4    # Higher mutation for exploration
  crossover_rate: 0.3   # Increased crossover for diversity
```

### LoRA Search Space (Optimized for Model Size)
```yaml
# For small models (270M-1B params)
lora_search_space:
  rank: [4, 8, 16, 32]                 # Lower ranks prevent overfitting
  alpha_multiplier: [1, 2]             # Alpha = rank * multiplier
  dropout: [0.0, 0.1]                  # Light dropout for regularization
  learning_rate: [5e-5, 1e-4, 2e-4]   # Conservative rates
  weight_decay: [0.01, 0.05, 0.1]     # Stronger regularization
  warmup_ratio: [0.05, 0.1, 0.15]     # Warmup helps stability
  max_grad_norm: [0.5, 1.0]           # Tighter gradient clipping
  use_rslora: [false, true]           # Rank-Stabilized LoRA
  target_modules_preset: ["standard", "extended"]

# For medium models (3B-7B params)
lora_search_space:
  rank: [8, 16, 32, 64]                # Balanced ranks
  learning_rate: [2e-5, 5e-5, 1e-4]   # More conservative rates
  weight_decay: [0.0, 0.01, 0.02]     # Moderate regularization
```

### Training Parameters (Based on Research)
```yaml
training:
  # For small models (270M)
  epochs_per_variant: 2   # 1-2 epochs to prevent overfitting
  batch_size: 16          # Moderate batch for better gradients

  # For medium models (4B)
  epochs_per_variant: 3   # 2-3 epochs for better convergence
  batch_size: 4           # Conservative for 4-bit quantization

  # Common settings
  gradient_accumulation_steps: 2-4    # Effective batch = 16-32
  eval_steps: 50-100                  # Frequent evaluation
  early_stopping_patience: 3          # Stop if no improvement
```

### GRPO Configuration
```yaml
grpo:
  enabled: false                # Enable via --grpo flag
  max_steps: 50-100             # Adjust based on model size
  temperature: 0.7-0.8          # Lower for focused generation
  pre_train_format: true        # Pre-train on format
  format_examples: 10-15        # Scale with model size
  pre_train_epochs: 1-2         # Minimal pre-training
  reward_weights:
    format: 1.0                 # Format compliance weight
    accuracy: 2.5-3.0           # Higher for task focus
    reasoning: 1.5-2.0          # Reasoning quality weight
```

## Supported Models

### Pre-configured Models
- **Qwen3-4B-Instruct**: 4B parameter model with 4-bit quantization support
- **Gemma3-270M-IT**: Lightweight 270M model, ideal for testing
- **Custom Models**: Any HuggingFace model compatible with Unsloth

### Adding Custom Models
Create a new config file in `EvoLoRA/config/` with your model settings.

## Datasets

### Built-in Support
- **GSM8K**: Math word problems (default)
- **Alpaca**: General instruction following
- **Dolly**: Diverse instruction dataset
- **Squad**: Question answering

### Custom Datasets
Datasets should have `question` and `answer` fields. For GRPO, optionally include `reasoning` field.

## Output Structure

```
lora_runs/
└── run_20250920_123456/
    ├── config/             # Run configuration
    ├── checkpoints/        # Generation checkpoints
    │   └── generations/    # Per-generation variants
    ├── models/
    │   ├── best/          # Best variant model
    │   └── variants/      # All variant models
    ├── logs/              # Training logs
    ├── history/           # Evolution history
    └── reports/           # Comparison reports
```

## Monitoring & Analysis

### During Training
- Real-time loss tracking
- Memory usage monitoring
- Per-variant performance metrics
- Generation-by-generation summaries

### Post-Training Analysis
```bash
# List recent runs
python -m EvoLoRA.cli_evolution list-runs

# Analyze evolution history
python -m EvoLoRA.cli_evolution analyze --run-dir lora_runs/run_20250920_123456

# Compare variants
python -m EvoLoRA.cli_evolution compare --run-dir lora_runs/run_20250920_123456
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Lower `max_seq_length`
- Use 4-bit quantization
- Reduce `population_size`

### Windows-Specific Issues
- Set `dataloader_num_workers: 0` in config
- Ensure Python is in PATH
- Run as Administrator if permission errors occur

### Unsloth Issues
```bash
# Reinstall with no cache
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir

# Verify installation
python -c "import unsloth; print('Unsloth OK')"
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for optimization technology
- [HuggingFace](https://huggingface.co/) for Transformers and PEFT
- [TRL](https://github.com/huggingface/trl) for training methods
- Model creators (Qwen, Google) for pre-trained models
