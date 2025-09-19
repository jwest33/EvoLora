@echo off
echo ========================================
echo Running Fast LoRA Evolution with GRPO
echo ========================================
echo.
echo This will run a quick evolutionary optimization using GRPO (Group Relative Policy Optimization)
echo for training reasoning and chain-of-thought models.
echo.
echo Configuration:
echo - GRPO training method for reasoning
echo - Smaller population size (3 variants)
echo - Reduced dataset (500 train, 100 eval)
echo - Reasoning format markers for chain-of-thought
echo - Optimized for GSM8K math reasoning dataset
echo.

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Set environment variable for GRPO
set USE_GRPO=true

REM Clear cache to ensure fresh start (optional, comment out if you want to keep cache)
REM echo Clearing cache...
REM if exist cache\datasets rmdir /s /q cache\datasets

REM Run with fast GRPO configuration
echo Starting fast evolution with GRPO...
python run_evolution.py evolve --config loralab/config/fast_grpo_config.yaml

echo.
echo GRPO Evolution complete!
pause
