# LoRALab Quick Start Guide

## Getting Started

Simply run the main launcher:

```bash
# Direct approach (no server required)
python run_auto_lora.py

# Or use the CLI directly
python -m loralab.cli evolve --config loralab/configs/documentation.yaml --use-direct
```

## Main Features

### 1. Evolution Training
Train LoRA adapters using co-evolution between Challenger (30B) and Solver (4B) models:
- **Quick test**: 5 generations (~5-10 minutes)
- **Standard run**: 20 generations (~20-30 minutes)
- **Full run**: 50 generations (~60-90 minutes)
- **Custom**: Configure your own parameters

### 2. Quality Comparison
Compare base model vs LoRA-enhanced outputs:
- **Quick visual**: Side-by-side comparison of 3 examples
- **Full analysis**: Detailed scoring of 5 examples with metrics

### 3. Test Existing Adapters
Test previously trained adapters on new code samples

### 4. Monitor Training
View real-time training metrics and progress

## Configuration

Edit `loralab/configs/documentation.yaml` to customize:
- `dataset_size_per_gen`: Tasks per generation (default: 20)
- `bootstrap_size`: Initial baseline tasks (default: 10)
- `generations`: Number of evolution cycles (default: 50)

## File Structure

```
lab/
├── run_auto_lora.py      # Main entry point
├── setup.py              # Package setup
├── requirements.txt      # Dependencies
├── loralab/
│   ├── cli.py            # Command-line interface
│   ├── core/             # LLM clients (direct & server)
│   ├── generation/       # Task generation (Challenger)
│   ├── adaptation/       # LoRA training (Solver)
│   ├── engine/           # Evolution orchestrator
│   ├── demos/            # Demo modules
│   ├── benchmarks/       # Performance benchmarks
│   ├── utils/            # Utilities
│   └── configs/          # Configuration files
├── examples/             # Example scripts
└── docs/                 # Documentation
    └── papers/           # Research papers
```

## Tips for Faster Testing

1. **Reduced dataset sizes** are configured by default (20 tasks/gen instead of 100)
2. **Bootstrap size** is only 10 tasks for quick baseline
3. **Progress tracking** shows ETA and tasks/minute
4. Use **Quick test** option for rapid iteration during development