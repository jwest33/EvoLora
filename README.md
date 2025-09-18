# LoRALab Self-Supervised Evolution System

LoRA adapter optimization using evolutionary algorithms - no teacher model required!

## Quick Start

```bash
# Install dependencies
pip install transformers peft datasets torch accelerate colorama

# For RTX 50-series GPUs (5060 Ti/5070/5080/5090), install PyTorch nightly (only tested on RTX 5060 Ti):
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Run evolution with default settings
python run_evolution.py evolve

# Quick test mode (2 generations, small dataset)
python run_evolution.py evolve --quick-test

# Resume from checkpoint
python run_evolution.py evolve --resume evolved_adapters/checkpoints/evolution_gen5.json
```

## GPU Support

```bash
# Install PyTorch nightly with CUDA 12.8 (see note above)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify installation
python -c "import torch; print(torch.cuda.get_arch_list())"
# Should include 'sm_120' in the output
```

### Configuration
Edit `loralab/config/config.yaml` to set device:
```yaml
model:
  device_map: "cuda:0"  # For GPU
  # device_map: "cpu"   # For CPU-only systems
```

## How It Works

The system uses **evolutionary optimization** to discover optimal LoRA configurations:

1. **Population Creation**: Generates 6 LoRA variants with different hyperparameters
2. **Training**: Each variant trains for 1 epoch on your dataset
3. **Evaluation**: Tests accuracy and perplexity on validation set
4. **Selection**: Top 2 performers survive to next generation
5. **Evolution**: Survivors mutate and crossover to create next generation
6. **Iteration**: Repeats for 10 generations to find optimal configuration

## Technical Details

### Fitness Scoring

Each LoRA variant is evaluated using a fitness function that combines accuracy and perplexity:

```python
fitness_score = accuracy - (perplexity / 100.0)
```

- **Accuracy**: Percentage of correct answers (0.0 to 1.0)
- **Perplexity**: Lower is better, divided by 100 for scaling
- Higher fitness scores indicate better performing variants

### Selection Process

The top 2 variants are selected as survivors based on:

1. **Sorting**: All variants in the population are sorted by fitness score (descending)
2. **Selection**: The top `keep_top` variants (default: 2) become survivors
3. **Tracking**: Best overall variant is tracked across all generations

```python
# From population.py
sorted_pop = sorted(population, key=lambda v: v.fitness_score(), reverse=True)
survivors = sorted_pop[:keep_top]
```

### Decision Logic for Evolution

**Why Top 2 Survivors?**
- Balances exploitation (keeping best) vs exploration (diversity)
- Prevents premature convergence to local optima
- Allows crossover between two different successful strategies

**When Best Variant Updates**:
- Checked after each generation completes
- Updates only if new variant has higher fitness than current best
- Best variant persisted across all generations (not reset)

**Convergence Detection**:
The system tracks convergence metrics but doesn't auto-stop:
- Monitors fitness improvement over last 5 generations
- Convergence detected if improvement < 1% over 5 generations
- Evolution continues for full 10 generations regardless
- This ensures thorough exploration of search space

### Evolution Mechanisms

#### Mutation
Each hyperparameter has a 30% chance (`mutation_rate`) to mutate independently:

**Rank Mutation** (30% chance):
- 50% probability: Slight adjustment (move to adjacent value in list)
- 50% probability: Random new value from [16, 32, 64, 128, 256]

**Learning Rate Mutation** (30% chance):
- 50% probability: Scale current LR by factor (0.5x, 0.8x, 1.25x, or 2.0x)
  - Result clamped to range [1e-6, 1e-2]
- 50% probability: Random new value from [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]

**Dropout Mutation** (30% chance):
- Always selects random new value from [0.05, 0.1, 0.15]

**Alpha Mutation** (30% chance):
- Randomly selects new multiplier from [1, 2, 4]
- Final alpha = rank × multiplier

#### Crossover
Two parent variants combine their hyperparameters:

```python
# Each hyperparameter has 50% chance to come from either parent
child.rank = random.choice([parent1.rank, parent2.rank])
child.learning_rate = random.choice([parent1.learning_rate, parent2.learning_rate])
child.dropout = random.choice([parent1.dropout, parent2.dropout])
```

#### Population Generation
For each new generation after the first:

1. **Elite Preservation**: Top 2 survivors are automatically included in next generation
2. **Offspring Creation**: Remaining 4 slots (for population_size=6) filled by:
   - **20% Crossover** (default `crossover_rate=0.2`): ~1 variant from crossover
   - **80% Mutation** (remaining slots): ~3 variants from mutation
3. **Parent Selection**:
   - For crossover: Two different survivors randomly selected
   - For mutation: Single survivor randomly selected (can be same parent multiple times)
4. **Fallback**: If only 1 survivor exists, all new variants created through mutation

### Evaluation Metrics

#### Accuracy Calculation
- Model generates answer for each validation question
- Answer checked using fuzzy matching:
  - Direct string containment
  - 80% word overlap threshold
  - Numerical value matching (for math problems)
  - Multiple choice letter extraction

#### Perplexity Calculation
- Measures model's confidence in correct answers
- Calculated as `exp(loss)` on the full question-answer pair
- Capped at 1000.0 to avoid infinities
- Lower perplexity indicates better language modeling

### Training Process

Each variant undergoes self-supervised training:

1. **Optimizer**: AdamW with variant-specific learning rate
2. **Batch Size**: 4 (configurable)
3. **Gradient Accumulation**: 2 steps (effective batch size: 8)
4. **Max Gradient Norm**: 1.0 for stability
5. **Epochs**: 1 per generation (configurable)

### Memory Management

The system carefully manages memory to support limited VRAM:

- Only one variant's model is loaded at a time
- Models are deleted and cache cleared between variants
- Checkpoints saved after each variant evaluation
- Base model shared across all variants (not duplicated)

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

# Pipeline test - minimal test to check for errors (3 variants, 2 generations, 15 examples)
python run_evolution.py evolve --pipeline-test

# Quick test - small but functional (2 variants, 2 generations, 120 examples)
python run_evolution.py evolve --quick-test

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

### Analyze Evolution History
```bash
# Analyze a specific run and generate visualizations
python run_evolution.py analyze --dir lora_runs/run_20250118_093000
```

### Manage Runs
```bash
# List all evolution runs
python run_evolution.py list-runs

# List runs in custom directory
python run_evolution.py list-runs --base-dir custom_runs

# Clean up old runs (keep last 5)
python run_evolution.py list-runs --cleanup

# Keep only last 3 runs
python run_evolution.py list-runs --cleanup --keep 3
```

This generates:
- **Family tree diagram**: Shows parent-child relationships and survival
- **Performance timeline**: Accuracy and perplexity trends
- **Hyperparameter heatmaps**: Effectiveness of parameter combinations
- **Survival analysis**: Which parameters led to survival
- **Mutation effectiveness**: Impact of different mutation types
- **Comprehensive report**: Markdown summary with all insights

## Evolution Algorithm Flow

### Complete Generation Cycle

1. **Generation 0 (Initial Population)**:
   - Creates 6 random variants from search space
   - Each variant gets unique ID: `r{rank}_lr{learning_rate:.0e}_d{dropout}_g{generation}`
   - No parents, all variants are independent

2. **Training Phase** (for each variant):
   - Load base model's weights
   - Apply LoRA configuration to create PEFT model
   - Train for 1 epoch on training dataset
   - Save checkpoint after training
   - Delete model to free memory (except last variant)

3. **Evaluation Phase** (for each variant):
   - Reload trained model from checkpoint
   - Evaluate on validation dataset:
     - Generate answers for each question
     - Calculate accuracy using fuzzy matching
     - Calculate perplexity on question-answer pairs
   - Compute fitness score: `fitness = accuracy - (perplexity / 100.0)`

4. **Selection Phase**:
   - Sort all variants by fitness score (descending)
   - Select top 2 variants as survivors
   - Track best variant across all generations

5. **Evolution Phase** (generations 1-9):
   - Keep 2 survivors in next generation (elitism)
   - Create 4 new variants:
     - ~1 from crossover (if 2+ survivors exist)
     - ~3 from mutation of random survivors
   - Increment generation counter

6. **Termination**:
   - After 10 generations complete
   - Best variant saved to `evolved_adapters/best_variant/`
   - Full history saved to `evolution_history.json`

### Example Evolution Trace

```
Generation 0: Random population
├── r16_lr1e-5_d0.05_g0:  fitness=-0.078 (acc=45.2%, ppl=12.3)
├── r32_lr2e-5_d0.10_g0:  fitness=-0.021 (acc=52.1%, ppl=10.1)  [SURVIVED]
├── r64_lr5e-5_d0.05_g0:  fitness=-0.032 (acc=48.0%, ppl=11.2)
├── r128_lr1e-4_d0.15_g0: fitness=-0.015 (acc=51.5%, ppl=10.5)  [SURVIVED]
├── r256_lr2e-4_d0.10_g0: fitness=-0.101 (acc=43.0%, ppl=13.1)
└── r16_lr5e-5_d0.15_g0:  fitness=-0.058 (acc=46.0%, ppl=11.8)

Generation 1: Evolved population
├── r32_lr2e-5_d0.10_g0:  [ELITE - kept from gen 0]
├── r128_lr1e-4_d0.15_g0: [ELITE - kept from gen 0]
├── r64_lr1e-4_d0.15_g1:  [CROSSOVER of top 2]
├── r32_lr1e-5_d0.10_g1:  [MUTATION of r32_lr2e-5]
├── r128_lr2e-4_d0.05_g1: [MUTATION of r128_lr1e-4]
└── r32_lr4e-5_d0.15_g1:  [MUTATION of r32_lr2e-5]

Generation 10: Converged
└── Best overall: r64_lr2e-5_d0.10_g7 with fitness=0.65 (73% accuracy)
```

## Output Structure

The new consolidated output structure organizes all runs under a base directory:

```
lora_runs/                      # Base directory for all runs
└── run_20250118_093000/       # Timestamped run directory
    ├── config/                 # Saved configuration
    │   └── config.yaml        # Complete config for this run
    ├── checkpoints/           # Evolution checkpoints
    │   ├── generations/       # Per-generation checkpoints
    │   │   ├── gen0/         # Generation 0 variants
    │   │   │   └── variant_id/
    │   │   │       └── config.json
    │   │   └── gen1/         # Generation 1 variants
    │   └── evolution_gen*.json  # Evolution state files
    ├── models/                # Saved model weights
    │   ├── best/             # Best variant found
    │   │   ├── adapter/      # LoRA weights
    │   │   └── config.json   # Configuration
    │   └── variants/         # All variant models
    │       └── variant_id/
    │           └── adapter/
    ├── reports/              # Evaluation and comparison reports
    │   ├── comparisons/      # Model comparison reports
    │   └── evaluations/      # Individual evaluations
    ├── analysis/             # Analysis outputs
    │   └── visualizations/   # Generated graphs and charts
    │       ├── family_tree.png
    │       ├── performance_timeline.png
    │       ├── hyperparameter_heatmap.png
    │       ├── survival_analysis.png
    │       ├── mutation_effectiveness.png
    │       └── evolution_analysis_report.md
    ├── history/              # Evolution history
    │   └── evolution_history.json
    └── logs/                 # Training and system logs
```

## Comparison Reports

After evolution completes, the system generates a detailed comparison report showing:

### What You'll See:
- **Improvements**: Questions the base model got wrong but LoRA got right
- **Regressions**: Questions the base model got right but LoRA got wrong
- **Both Wrong**: Challenging questions neither model could answer
- **Accuracy Breakdown**: Detailed statistics on model performance

Reports are automatically generated for the best variant and saved in `evolved_adapters/evaluation_reports/`.

## Hyperparameters and Search Space

### LoRA Configuration Details

**Target Modules**:
The system applies LoRA to the following attention modules:
- `q_proj`: Query projection layer
- `k_proj`: Key projection layer
- `v_proj`: Value projection layer
- `o_proj`: Output projection layer

These are the standard attention layers in transformer models where LoRA is most effective.

**Hyperparameter Search Space**:
- **Rank**: [16, 32, 64, 128, 256]
  - Controls capacity of LoRA adapter
  - Higher rank = more parameters = more capacity
- **Alpha**: rank × [1, 2, 3]
  - Scaling factor for LoRA updates
  - Higher alpha = stronger LoRA influence
- **Learning Rate**: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
  - Controls optimization step size
  - Critical for training stability
- **Dropout**: [0.05, 0.1, 0.15]
  - Regularization during training
  - Prevents overfitting on small datasets

### Training Configuration

**Optimizer Settings**:
- Optimizer: AdamW
- Weight Decay: 0.01
- Gradient Clipping: 1.0 (max norm)
- Warmup Ratio: 0.1 (10% of training steps)

**Batch Configuration**:
- Per-device Batch Size: 4
- Gradient Accumulation: 16 steps
- Effective Batch Size: 64
- Mixed Precision: FP16 enabled

### Variant Identification System

Each variant has a unique ID encoding its configuration:
```
r{rank}_lr{learning_rate}_d{dropout}_g{generation}
```

Example: `r64_lr2e-5_d0.10_g3`
- Rank: 64
- Learning Rate: 2e-5
- Dropout: 0.10
- Generation: 3

## Example Available Datasets

- `mmlu-pro`: Multi-task language understanding
- `gsm8k`: Grade school math problems
- `squad`: Question answering
- `alpaca`: Instruction following
- `dolly`: Diverse instruction tasks

## Testing Modes

### Pipeline Test Mode
For testing the entire pipeline for errors without waiting for full training:

```bash
python run_evolution.py evolve --pipeline-test
```

**Pipeline test configuration:**
- Population: 3 variants (minimum to test crossover + mutation)
- Generations: 2 (minimum to test evolution mechanics)
- Training examples: 10
- Evaluation examples: 5
- Ranks: Limited to [16, 32] for speed
- Total runtime: ~5-10 minutes

**What it tests:**
- Model loading and LoRA initialization
- Training loop execution
- Evaluation metrics calculation
- Survivor selection (2 from 3 variants)
- Crossover between 2 survivors
- Mutation of survivors
- Checkpoint saving and loading
- Best variant tracking

### Quick Test Mode
For a slightly larger but still fast test:

```bash
python run_evolution.py evolve --quick-test
```

- Population: 2 variants
- Generations: 2
- Training examples: 100
- Evaluation examples: 20
- Runtime: ~15-20 minutes

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
# Resume from specific generation checkpoint
python run_evolution.py evolve --resume evolved_adapters/checkpoints/evolution_gen5.json

# Checkpoint contains:
# - Current generation number
# - Surviving variants with all hyperparameters
# - Best variant found so far
# - Complete evolution history
```

### Checkpoint Recovery Process

When resuming from checkpoint:
1. Loads surviving variants from previous generation (configurations only)
2. Restores best variant tracking
3. Continues evolution from next generation
4. Preserves all historical metrics

**Checkpoint Storage Strategy**:
- **Evolution checkpoints** (`evolution_gen*.json`): Only configurations and metrics (~1-2 KB each)
- **Variant models** (`gen*/*/adapter/`): Full trained LoRA weights (~5 MB for rank 128)
- **Best variant** (`best_variant/adapter/`): Complete model saved for deployment

**Storage Requirements (fair warning, this is probably a wildly inaccuracy estimation)**:
- Each LoRA adapter: 0.6-10 MB depending on rank (rank 128 = ~5 MB)
- Total for 60 variants: ~300 MB (rank 128) to ~600 MB (rank 256)
- Evolution JSON files: < 1 MB total
