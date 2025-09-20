@echo off
setlocal enabledelayedexpansion

REM Check if GRPO flag is provided
set USE_GRPO=0
if /i "%1"=="--gpro" set USE_GRPO=1
if /i "%1"=="gpro" set USE_GRPO=1
if /i "%1"=="--grpo" set USE_GRPO=1
if /i "%1"=="grpo" set USE_GRPO=1

REM Set environment variables
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
set UNSLOTH_RETURN_LOGITS=1
set TOKENIZERS_PARALLELISM=false

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Clear screen for clean output
cls

REM Run evolution with appropriate config and let Python handle all formatting
if !USE_GRPO!==1 (
    python run_evolution.py evolve --config loralab/config/gemma_config.yaml --grpo
) else (
    python run_evolution.py evolve --config loralab/config/gemma_config.yaml
)

echo.
pause
