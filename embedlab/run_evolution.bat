@echo off
REM === R-Zero Evolution System Demo ===

echo Starting R-Zero Self-Evolving Reasoning System...
echo.

REM === Ensure virtual environment ===
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM === Activate virtual environment ===
call .venv\Scripts\activate

REM === Install/Update project ===
echo Installing dependencies...
pip install -e . -q

REM === Create output directory ===
if not exist evolution_runs mkdir evolution_runs

REM === Run evolution ===
echo.
echo Starting evolution process...
echo LLM Server: http://localhost:8000
echo Embedding Server: http://localhost:8002
echo.

embedlab-evolve evolve ^
  --hierarchy data\sample_hierarchy.yaml ^
  --output evolution_runs\rzero_experiment ^
  --generations 20 ^
  --population 10 ^
  --llm http://localhost:8000 ^
  --embedding http://localhost:8002 ^
  --checkpoint 5 ^
  --verbose

echo.
echo Evolution complete! Results saved to evolution_runs\rzero_experiment
pause
