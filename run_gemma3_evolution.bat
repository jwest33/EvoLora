@echo off
echo.
echo ================================================================
echo           GEMMA3-270M GRPO EVOLUTIONARY OPTIMIZATION
echo ================================================================
echo.

REM Set environment variables for GRPO
set USE_GRPO=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Starting evolutionary GRPO optimization on Gemma3-270M...
echo.
echo Configuration: loralab/config/gemma3_grpo_config.yaml
echo Model: Gemma3-270M-IT (unsloth/gemma-3-270m-it)
echo Method: GRPO (Group Relative Policy Optimization)
echo.

REM Run evolution with Gemma3 GRPO config
python run_evolution.py evolve --config loralab/config/gemma3_grpo_config.yaml

echo.
echo Gemma3 GRPO Evolution complete!
pause