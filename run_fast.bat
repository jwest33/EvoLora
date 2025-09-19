@echo off
echo ========================================
echo Running Fast LoRA Evolution
echo ========================================
echo.
echo This will run a quick evolutionary optimization with:
echo - Smaller population size (3 variants)
echo - Reduced dataset (500 train, 100 eval)
echo - Faster evaluation (perplexity only)
echo - Optimized batch sizes
echo.

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Clear cache to ensure fresh start (optional, comment out if you want to keep cache)
REM echo Clearing cache...
REM if exist cache\datasets rmdir /s /q cache\datasets

REM Run with fast configuration
echo Starting fast evolution...
python run_evolution.py evolve --config loralab/config/fast_config.yaml

echo.
echo Evolution complete!
pause
