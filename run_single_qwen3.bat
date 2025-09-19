@echo off
echo.
echo ================================================================
echo                SINGLE GRPO TRAINING - QWEN3-4B
echo ================================================================
echo.

REM Set environment variables for GRPO
set USE_GRPO=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Starting single GRPO training on Qwen3-4B...
echo.

REM Run single GRPO training with Qwen3
python run_single_grpo.py ^
    --model qwen3 ^
    --rank 16 ^
    --learning-rate 1e-4 ^
    --batch-size 4 ^
    --max-steps 50

echo.
echo Single GRPO training complete!
pause