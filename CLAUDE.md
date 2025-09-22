# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an R-Zero inspired self-evolving reasoning system that implements a Teacher-Solver co-evolution approach. The system generates its own training data from scratch and uses GRPO (Group Relative Policy Optimization) to teach models step-by-step reasoning.

## Commands

### Virtual Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/MacOS
```

### Running the R-Zero Training Pipeline

```bash
# Run complete R-Zero evolution (Teacher + Solver with GRPO)
python run_rzl.py

# Test GRPO training with small dataset
python test_grpo.py

# Test Teacher problem generation only
python test_teacher.py
```

### Model Export and Inference

```bash
# Export trained model to GGUF format for llama.cpp
python export_to_gguf.py                     # Auto-detect latest checkpoint
python export_to_gguf.py --quantization q6_k # Specific quantization

# Chat with exported GGUF model
python chat_cli.py                     # Auto-find model
python chat_cli.py --model outputs/gguf/rzero_solver_q8_0.gguf

# Setup llama.cpp tools (if needed for GGUF conversion)
python setup_llama_cpp.py
```

## Architecture

### Core R-Zero Implementation (`run_rzl.py`)

The system implements two co-evolving agents:

**TeacherAgent**:
- Uses llama.cpp with Qwen3-30B GGUF model (`C:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf`)
- Generates math problems at varying difficulty levels
- Evolves its prompt based on Solver performance (target: 60-70% accuracy)
- Saves/loads state for continuous improvement

**SolverAgent**:
- Uses Unsloth FastModel with Gemma-3-1B (4-bit quantized)
- Learns reasoning through GRPO training with custom reward functions
- Reward components: format matching, answer correctness, number extraction
- Uses special reasoning format markers: `<start_working_out>`, `<SOLUTION>`

### Key Technical Details

**GRPO Training Configuration (Optimized)**:
- Batch size: 4 per device
- Gradient accumulation: 1 step
- Number of generations: 4 (total effective batch = 16)
- Learning rate: 5e-6 with cosine schedule
- Max prompt length: 128 tokens
- Max completion length: 150 tokens
- LoRA configuration: r=64, alpha=16, dropout=0, target all linear layers
- Reward functions: format matching, answer correctness, length penalty

**Chat Template** (`qwen3_nonthinking.jinja`):
- Custom Jinja2 template for Qwen3 models
- Adds math problem formatting with `\boxed{}` for answers
- Includes step-by-step calculation format with `<<>>` brackets

**Model Paths**:
- Teacher model: `C:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf`
- Solver checkpoints saved to: `outputs/solver/iteration_N`
- GGUF exports saved to: `outputs/gguf/`

### Dependencies

**Minimal Dependencies for R-Zero Pipeline**:
- `rzl.utils.cli_formatter` - The ONLY rzl module needed (provides CLIFormatter, SpinnerProgress)
- All other dependencies are external packages

**Key External Packages**:
- `unsloth` - For efficient model training with LoRA
- `transformers` - Model loading and tokenization
- `trl` - For GRPO trainer implementation
- `llama-cpp-python` - For loading GGUF models
- `datasets` - For dataset handling
- `torch` - PyTorch for GPU acceleration
- `colorama` - For colored terminal output

### File Structure

```
lab/
├── run_rzl.py                 # Main R-Zero implementation
├── test_grpo.py              # Test GRPO training
├── test_teacher.py           # Test Teacher generation
├── export_to_gguf.py         # Export to GGUF format
├── chat_cli.py         # Interactive chat interface
├── setup_llama_cpp.py        # Setup llama.cpp tools
├── qwen3_nonthinking.jinja   # Chat template for Qwen3
├── rzl/
│   └── utils/
│       └── cli_formatter.py  # CLI formatting utilities (synthwave colors)
├── outputs/
│   ├── solver/               # Training checkpoints
│   └── gguf/                 # Exported GGUF models
└── tools/
    └── llama.cpp/            # llama.cpp installation (for GGUF conversion)
```

## Training Flow (Optimized Settings)

1. **Pre-training**:
   - Dataset: 30 GSM8K examples
   - Steps: 10 (warm-up to learn format)

2. **Evolution Loop** (8 iterations recommended):
   - Teacher generates 20 + (iteration × 5) problems per iteration
   - Solver trains for 75 GRPO steps per iteration
   - Evaluate Solver accuracy on 10 test problems
   - Teacher evolves prompt based on performance
   - Adjust difficulty (increase if >70% accuracy, decrease if <40%)

3. **Export**: Convert best checkpoint to GGUF for efficient inference

## Important Configuration

**Always ask the user to run tests**
- The virtual environment is hard to access, always ask the user to run tests

**GPU Memory Requirements**:
- Teacher (llama.cpp): ~6GB VRAM for Q6_K quantized 30B model
- Solver (Unsloth): ~4GB VRAM for 4-bit Gemma-3-1B with LoRA

**Windows-Specific Notes**:
- Use `tools\llama.cpp` for GGUF conversion tools
- Convert script is `convert_hf_to_gguf.py` (with underscores, not hyphens)
- Quantize executable may be `llama-quantize.exe` or `quantize.exe`

## Debugging

**Common Issues**:
- xformers dtype mismatch: Mixed float16/bfloat16 tensors (usually ignorable warning)
- GGUF export fails: Run `python setup_llama_cpp.py` to install conversion tools
- SpinnerProgress context manager: Fixed in latest cli_formatter.py
- Reward values showing 0: Check that GRPO logs contain 'reward' and 'rewards/' keys
