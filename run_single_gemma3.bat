@echo off
echo.
echo ================================================================
echo                SINGLE GRPO TRAINING - GEMMA3-270M
echo ================================================================
echo.

REM Set environment variables for GRPO
set USE_GRPO=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Starting single GRPO training on Gemma3-270M...
echo.

REM Run single GRPO training with Gemma3
python run_single_grpo.py ^
    --model gemma3 ^
    --rank 128 ^
    --learning-rate 2e-4 ^
    --batch-size 8 ^
    --max-steps 100

echo.
echo Single GRPO training complete!
pause