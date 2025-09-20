@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo                    QWEN MODEL EVOLUTION
echo ================================================================
echo.

REM Check if GRPO flag is provided
set USE_GRPO=0
if /i "%1"=="grpo" (
    set USE_GRPO=1
    echo Mode: GRPO (Group Relative Policy Optimization)
) else if /i "%1"=="GRPO" (
    set USE_GRPO=1
    echo Mode: GRPO (Group Relative Policy Optimization)
) else (
    echo Mode: Standard SFT (Supervised Fine-Tuning)
)

REM Set environment variables
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
set UNSLOTH_RETURN_LOGITS=1
set TOKENIZERS_PARALLELISM=false

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo.
echo Model: Qwen3-4B-Instruct
echo Starting evolutionary optimization...
echo.

REM Run evolution with appropriate config
if !USE_GRPO!==1 (
    echo Using GRPO configuration...
    python run_evolution.py evolve --config loralab/config/qwen_config.yaml --grpo
) else (
    echo Using standard SFT configuration...
    python run_evolution.py evolve --config loralab/config/qwen_config.yaml
)

echo.
echo Evolution complete!
pause