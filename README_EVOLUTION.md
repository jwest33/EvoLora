# LoRALab Self-Supervised Evolution System

LoRA adapter optimization using evolutionary algorithms - no teacher model required!

## Quick Start

```bash
# Install dependencies
pip install transformers peft datasets torch accelerate

# Run evolution with default settings
python run_evolution.py evolve

# Quick test mode (2 generations, small dataset)
python run_evolution.py evolve --quick-test

# Resume from checkpoint
python run_evolution.py evolve --resume evolved_adapters/checkpoints/evolution_gen5.json
```

## How It Works

The system uses **evolutionary optimization** to discover optimal LoRA configurations:

1. **Population Creation**: Generates 6 LoRA variants with different hyperparameters
2. **Training**: Each variant trains for 1 epoch on your dataset
3. **Evaluation**: Tests accuracy and perplexity on validation set
4. **Selection**: Top 2 performers survive to next generation
5. **Evolution**: Survivors mutate and crossover to create next generation
6. **Iteration**: Repeats for 10 generations to find optimal configuration

## Key Benefits

- **No Teacher Required**: Saves 23GB+ RAM by eliminating teacher model
- **Automatic Optimization**: Evolution finds best rank, learning rate, dropout
- **Memory Efficient**: Only loads one 4B model at a time
- **Crash Resilient**: Checkpoints after each generation
- **Fast Iteration**: 1 epoch per variant for rapid evolution

## Configuration

The system is configured via `loralab/config/config.yaml`:

```yaml
mode: "self_supervised"

self_supervised:
  model:
    path: "Qwen/Qwen3-4B-Instruct-2507"  # Your model

  evolution:
    population_size: 6      # Variants per generation
    generations: 10         # Evolution cycles
    keep_top: 2            # Survivors per generation

  lora_search_space:
    rank: [16, 32, 64, 128, 256]           # LoRA ranks to explore
    learning_rate: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]  # LR options
    dropout: [0.05, 0.1, 0.15]             # Dropout rates

  dataset:
    sources: ["mmlu-pro"]   # Dataset to use
    train_size: 10000       # Training examples
    eval_size: 1000         # Validation examples
```

## Commands

### Run Evolution
```bash
# Full evolution (10 generations)
python run_evolution.py evolve

# Custom settings
python run_evolution.py evolve --generations 20 --population 10

# Specific dataset size
python run_evolution.py evolve --train-size 5000 --eval-size 500
```

### Evaluate Adapter
```bash
# Test a specific adapter
python run_evolution.py evaluate --adapter evolved_adapters/best_variant/adapter

# Compare base model vs LoRA adapter with detailed report
python run_evolution.py evaluate --adapter evolved_adapters/best_variant/adapter --compare

# Custom dataset
python run_evolution.py evaluate --adapter path/to/adapter --dataset gsm8k --compare
```

### List Datasets
```bash
# See available datasets
python run_evolution.py list-datasets
```

## Evolution Process

Generation progression example:
```
Generation 0: Random population
├── r16_lr1e-5:  accuracy=45%, perplexity=12.3
├── r32_lr2e-5:  accuracy=52%, perplexity=10.1  [SURVIVED]
├── r64_lr5e-5:  accuracy=48%, perplexity=11.2
├── r128_lr1e-4: accuracy=51%, perplexity=10.5  [SURVIVED]
├── r256_lr2e-4: accuracy=43%, perplexity=13.1
└── r16_lr5e-5:  accuracy=46%, perplexity=11.8

Generation 5: Converging on optimal
├── r32_lr2e-5:  accuracy=68%, perplexity=7.2   [SURVIVED]
├── r64_lr2e-5:  accuracy=71%, perplexity=6.8   [SURVIVED]
├── r32_lr1e-5:  accuracy=65%, perplexity=7.9
└── ...

Generation 10: Optimal found
└── Best: r64_lr2e-5 with 73% accuracy
```

## Output Structure

```
evolved_adapters/
├── best_variant/           # Best adapter found
│   ├── adapter/           # LoRA weights
│   └── config.json        # Configuration
├── checkpoints/           # Generation checkpoints
│   ├── gen0/             # All variants from generation 0
│   ├── gen1/
│   └── evolution_gen*.json
├── evaluation_reports/    # Detailed comparison reports
│   ├── comparison_report_*.md  # Human-readable reports
│   └── comparison_report_*.json # Machine-readable data
└── evolution_history.json # Full evolution history
```

## 📊 Comparison Reports

After evolution completes, the system generates a detailed comparison report showing:

### What You'll See:
- **Improvements**: Questions the base model got wrong but LoRA got right
- **Regressions**: Questions the base model got right but LoRA got wrong
- **Both Wrong**: Challenging questions neither model could answer
- **Accuracy Breakdown**: Detailed statistics on model performance

### Example Report Output:
```markdown
## 🎯 Improvements (Base Wrong → LoRA Correct)

### Example 1
**Question**: What is the derivative of x^3?
**Correct Answer**: 3x^2
**Base Model Answer** ❌: 2x^2
**LoRA Model Answer** ✅: 3x^2

### Example 2
**Question**: Which planet is known as the Red Planet?
**Correct Answer**: Mars
**Base Model Answer** ❌: Jupiter
**LoRA Model Answer** ✅: Mars
```

Reports are automatically generated for the best variant and saved in `evolved_adapters/evaluation_reports/`.

## Hyperparameters Explored

The system automatically explores:
- **LoRA Rank**: 16, 32, 64, 128, 256
- **Learning Rate**: 1e-5 to 2e-4
- **Dropout**: 0.05, 0.1, 0.15
- **Alpha**: Automatically scaled with rank

## Example Available Datasets

- `mmlu-pro`: Multi-task language understanding
- `gsm8k`: Grade school math problems
- `squad`: Question answering
- `alpaca`: Instruction following
- `dolly`: Diverse instruction tasks

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in config
training:
  batch_size: 2  # Reduce from 4
```

### Slow Training
```bash
# Reduce population or dataset size
python run_evolution.py evolve --population 4 --train-size 5000
```

### Resume After Crash
```bash
# Automatically resumes from last checkpoint
python run_evolution.py evolve --resume
```
