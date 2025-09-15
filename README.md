# R-Zero Self-Evolving Reasoning System

## Quick Start

```bash
# 1. Install the package
pip install -e .

# 2. Start your servers (LLM and Embedding)
# Default: http://localhost:8000 and http://localhost:8002

# 3. Run evolution with automatic dataset management
embedlab-evolve evolve --hierarchy data/sample_hierarchy.yaml --output evolution_runs/my_experiment

# 4. Monitor progress
embedlab-evolve monitor --dir checkpoints
```

## Overview

This project is inspired by the **R-Zero** paper ([R-Zero: Self-Evolving Reasoning LLM from Zero Data](2508_05004v2.pdf)), which introduces a self-evolving reasoning system that generates its own training data and improves through co-evolution of Challenger and Solver agents.

## Key Concepts

### 1. **Zero-Data Bootstrap**
- Starts with only a hierarchy definition (YAML)
- Uses LLM to generate initial training examples
- No manual dataset creation required

### 2. **Challenger-Solver Co-Evolution**
- **Challenger Agent**: Generates increasingly difficult routing queries
  - Creates boundary cases between categories
  - Produces ambiguous and misleading queries
  - Adjusts difficulty based on Solver performance

- **Solver Agent**: Attempts to route queries using evolved instructions
  - Tests multiple instruction variants
  - Uses embeddings for similarity-based routing
  - Tracks failure patterns for improvement

### 3. **Instruction Evolution**
- Population of instruction variants evolve through genetic algorithm
- Mutation: LLM-guided instruction modifications
- Crossover: Combining successful instruction elements
- Selection: Fitness based on routing accuracy

## Prerequisites

### Required Servers

You need two servers running before using the application:

1. **LLM Server** (Language Model)
   - Default URL: `http://localhost:8000`
   - Recommended: Qwen3-4B or similar
   - Used for: Generating queries, evolving instructions, routing decisions

2. **Embedding Server** (Text Embeddings)
   - Default URL: `http://localhost:8002`
   - Recommended: Qwen3-Embedding-0.6B or similar
   - Used for: Computing text similarity for routing

**Using Custom Server URLs:**
```bash
embedlab-evolve evolve \
  --llm http://192.168.1.100:8000 \
  --embedding http://192.168.1.100:8002 \
  ...
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd lab

# Install in development mode
pip install -e .

# Verify installation
embedlab-evolve --help
```

## How to Use the Application

### Step-by-Step Guide

#### 1. Choose Your Task Goal

First, decide what type of task you want to optimize instructions for:

```bash
# List available task goals
embedlab-evolve list-goals
```

Available goals:
- `hierarchical_routing` - Route queries through category trees
- `semantic_search` - Find semantically relevant documents
- `intent_classification` - Classify user intent
- `question_answering` - Retrieve answer passages

#### 2. Prepare Your Hierarchy

Create a YAML file defining your category structure:

```yaml
# data/my_hierarchy.yaml
name: root
description: Root of the hierarchy
children:
  - name: Category1
    description: Description of category 1
    children:
      - name: Subcategory1A
        description: Specific subcategory
      - name: Subcategory1B
        description: Another subcategory
  - name: Category2
    description: Description of category 2
```

#### 3. Run Evolution

**Basic Usage (Automatic Dataset Management):**
```bash
embedlab-evolve evolve \
  --hierarchy data/my_hierarchy.yaml \
  --output runs/experiment1 \
  --generations 50 \
  --population 20
```

The system will:
  :heavy_check_mark: Check for existing compatible datasets
  :heavy_check_mark: Reuse them if they match your task goal
  :heavy_check_mark: Generate new ones only if needed
  :heavy_check_mark: Save best instructions and results

**Force New Dataset Generation:**
```bash
embedlab-evolve evolve \
  --hierarchy data/my_hierarchy.yaml \
  --output runs/experiment2 \
  --regenerate  # Forces new dataset creation
```

#### 4. Monitor Progress

```bash
# Watch evolution in real-time
embedlab-evolve monitor --dir runs/experiment1/checkpoints
```

#### 5. Use the Results

After evolution completes, find your optimized instruction in:
- `runs/experiment1/best_instruction.txt` - The optimized instruction
- `runs/experiment1/evolution_results.json` - Full results and metrics

### Quick Start Scripts

**Windows:**
```bash
# Run with interactive menu
run_evolution_examples.bat

# Run default evolution
run_evolution.bat
```

## Complete Command Reference

### `evolve` - Run Evolution Process

```bash
embedlab-evolve evolve \
  --hierarchy data/sample_hierarchy.yaml \
  --output evolution_runs/experiment1 \
  --generations 50 \
  --population 20 \
  --task-goal hierarchical_routing \
  --llm http://localhost:8000 \
  --embedding http://localhost:8002

# Force dataset regeneration (by default, existing datasets are reused)
embedlab-evolve evolve \
  --regenerate \
  ...
```

**Dataset Reuse:**
- The system automatically checks for existing datasets in the output directory
- If datasets exist and match the current task goal, they are reused (saves time)
- Use `--regenerate` flag to force creation of new datasets
- Dataset metadata tracks task goal, generation time, and parameters

**Task Goal Options:**
- `hierarchical_routing` - Route queries through category hierarchy (default)
- `semantic_search` - Find semantically relevant documents
- `intent_classification` - Classify user intent
- `question_answering` - Retrieve answer passages
- `custom` - Define your own task goal

**Custom Task Goals:**
```bash
# Using inline custom goal
embedlab-evolve evolve \
  --task-goal custom \
  --custom-objective "Find relevant API documentation" \
  --custom-instruction "Locate API docs that answer the developer's question" \
  ...

# Using YAML file
embedlab-evolve evolve \
  --task-goal-file embedlab/config/custom_goals.yaml \
  --task-goal technical_documentation_search \
  ...
```

### `generate-dataset` - Generate Dataset Only

```bash
embedlab-evolve generate-dataset \
  --hierarchy data/sample_hierarchy.yaml \
  --output data/synthetic_dataset.csv \
  --size 10 \
  --augment 3
```

### `test-routing` - Test Routing Performance

```bash
embedlab-evolve test-routing \
  --hierarchy data/sample_hierarchy.yaml \
  --dataset data/test_dataset.csv \
  --instruction "Route to the most relevant category based on key terms and intent."
```

### `monitor` - Monitor Progress

```bash
embedlab-evolve monitor \
  --dir checkpoints
```

### `list-goals` - List Available Task Goals

```bash
embedlab-evolve list-goals
```

## How Evolution Works

### Generation Loop

1. **Challenger Phase**
   - Generates batch of difficult queries
   - Adjusts difficulty based on Solver performance
   - Creates boundary, ambiguous, and misleading examples

2. **Solver Phase**
   - Tests each instruction variant on new queries
   - Routes using embedding similarity
   - Tracks success/failure patterns

3. **Evolution Phase**
   - Calculates fitness for each instruction
   - Applies genetic operations (mutation, crossover)
   - Generates targeted mutations for problem areas

4. **Progress Tracking**
   - Updates difficulty curriculum
   - Saves checkpoints
   - Records metrics and history

### Reward System

**Challenger Rewards:**
- Optimal when Solver achieves 60-70% accuracy
- Penalized for too easy or too hard queries
- Bonus for query diversity

**Solver Rewards:**
- Direct reward for routing accuracy
- Bonus for high confidence on correct predictions
- Reward for continuous improvement

## Configuration

Key parameters in `evolution_config.py`:

```python
config = EvolutionConfig(
    # Task Goal
    task_goal_name="hierarchical_routing",  # Or any predefined goal

    # Evolution
    generations=100,              # Number of evolution cycles
    population_size=20,           # Instruction variants

    # Genetic Algorithm
    mutation_rate=0.2,           # Probability of mutation
    crossover_rate=0.5,          # Probability of crossover
    elite_size=4,                # Top performers to keep

    # Difficulty
    initial_difficulty=0.3,      # Starting difficulty
    difficulty_increment=0.05,   # Difficulty adjustment rate

    # Data Generation
    dataset_size_per_gen=50,     # Queries per generation
    seed_examples_per_node=5,    # Initial examples per category
)
```

### Task Goal Configuration

The system now supports configurable task goals that define:
- **Objective**: What the instruction should achieve
- **Base Instruction**: Starting point for evolution
- **Instruction Styles**: Variations to explore
- **Mutation Strategies**: How to modify instructions
- **Success Criteria**: Target performance metrics

#### Creating Custom Task Goals

Edit `embedlab/config/custom_goals.yaml`:

```yaml
my_custom_task:
  name: "My Custom Retrieval Task"
  type: "custom"
  description: "Description of what this task does"
  objective: "Main goal to achieve"
  base_instruction: "Starting instruction text"

  instruction_styles:
    - "style1"
    - "style2"

  mutation_strategies:
    - "strategy1"
    - "strategy2"

  success_criteria:
    accuracy: 0.85
    precision: 0.8
```

## Output Files

The system generates:

1. **Best Instruction** (`best_instruction.txt`)
   - Optimized routing instruction
   - Ready for production use

2. **Evolution Results** (`evolution_results.json`)
   - Final test accuracy
   - Evolution history
   - Top performing instructions
   - Confusion matrix

3. **Checkpoints** (`checkpoints/`)
   - Periodic saves of evolution state
   - Population snapshots
   - Progress metrics

4. **Datasets** (`train_data.csv`, `test_data.csv`)
   - Generated synthetic examples
   - Balanced across categories

5. **Dataset Metadata** (`dataset_metadata.json`)
   - Task goal information
   - Generation timestamp
   - Configuration parameters

## Troubleshooting

### Common Issues and Solutions

#### Servers not responding
```bash
# Check if servers are accessible
curl http://localhost:8000/health
curl http://localhost:8002/health

# Use custom server URLs if needed
embedlab-evolve evolve --llm http://your-server:8000 ...
```

#### Dataset regeneration issues
```bash
# Force regeneration if dataset is corrupted
embedlab-evolve evolve --regenerate ...

# Check dataset status
python test_dataset_reuse.py runs/experiment1
```

#### Low accuracy
- Increase population size: `--population 30`
- Run more generations: `--generations 100`
- Try different task goal: `--task-goal semantic_search`

#### Slow evolution
- Reduce dataset size per generation in config
- Use smaller population: `--population 10`
- Check server response times

#### Memory issues
- Reduce batch sizes in configuration
- Use smaller embedding models
- Limit dataset size

## Tips for Best Results

1. **Start Small**: Begin with 20-30 generations to test your setup
2. **Reuse Datasets**: Let the system reuse datasets to save time
3. **Monitor Progress**: Use the monitor command to track fitness improvement
4. **Adjust Difficulty**: The system automatically adjusts difficulty based on performance
5. **Save Checkpoints**: Use `--checkpoint 10` to save progress every 10 generations

### Custom Task Goals
```bash
# Edit embedlab/config/custom_goals.yaml to define your task
embedlab-evolve evolve \
  --task-goal-file embedlab/config/custom_goals.yaml \
  --task-goal my_custom_task \
  --output runs/custom
```

## Architecture

```
embedlab/
├── core/                    # Infrastructure
│   ├── llm_client.py       # LLM server interface
│   └── embedding_client.py # Embedding server interface
├── generation/             # Data generation
│   ├── challenger.py       # Challenger agent
│   └── dataset_builder.py  # Dataset orchestration
├── evolution/              # Instruction evolution
│   ├── solver.py           # Solver agent
│   └── instruction_optimizer.py # Genetic algorithm
├── engine/                 # Main loop
│   ├── evolution_loop.py   # Orchestration
│   └── reward_calculator.py # Reward functions
└── config/                 # Configuration
    ├── evolution_config.py # System parameters
    ├── task_goals.py       # Task goal definitions
    └── custom_goals.yaml   # Custom task goals
```

## Key Features

1. **Self-Driven Learning**: No manual data annotation required
2. **Adaptive Curriculum**: Difficulty automatically adjusts to learning progress
3. **Targeted Evolution**: Mutations specifically address observed failures
4. **Co-Evolution**: Challenger and Solver improve together
5. **Instruction Discovery**: Finds optimal phrasing through evolution
6. **Task-Aware Generation**: Datasets adapt to specific task goals
7. **Smart Dataset Reuse**: Saves time by reusing compatible datasets

## Citation

Based on the R-Zero paper:
```
@misc{huang2025rzeroselfevolvingreasoningllm,
      title={R-Zero: Self-Evolving Reasoning LLM from Zero Data}, 
      author={Chengsong Huang and Wenhao Yu and Xiaoyang Wang and Hongming Zhang and Zongxia Li and Ruosen Li and Jiaxin Huang and Haitao Mi and Dong Yu},
      year={2025},
      eprint={2508.05004},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.05004}, 
}
```
