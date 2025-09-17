@echo off
REM === R-Zero Evolution System Examples ===

echo R-Zero Self-Evolving Reasoning System - Example Commands
echo ==========================================================
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

echo.
echo Choose an example to run:
echo.
echo 1. Hierarchical Routing (default - uses existing dataset if available)
echo 2. Semantic Search Task (uses existing dataset if available)
echo 3. Intent Classification Task (uses existing dataset if available)
echo 4. Force Dataset Regeneration (hierarchical routing with new dataset)
echo 5. Custom Task Goal from YAML file
echo 6. List Available Task Goals
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto hierarchical
if "%choice%"=="2" goto semantic
if "%choice%"=="3" goto intent
if "%choice%"=="4" goto regenerate
if "%choice%"=="5" goto custom
if "%choice%"=="6" goto list
goto invalid

:hierarchical
echo.
echo Running Hierarchical Routing Evolution (will reuse existing dataset if available)...
echo.
embedlab-evolve evolve ^
  --hierarchy data\sample_hierarchy.yaml ^
  --output evolution_runs\hierarchical_routing ^
  --task-goal hierarchical_routing ^
  --generations 20 ^
  --population 10 ^
  --llm http://localhost:8000 ^
  --embedding http://localhost:8002 ^
  --checkpoint 5 ^
  --verbose
goto end

:semantic
echo.
echo Running Semantic Search Evolution (will reuse existing dataset if available)...
echo.
embedlab-evolve evolve ^
  --hierarchy data\sample_hierarchy.yaml ^
  --output evolution_runs\semantic_search ^
  --task-goal semantic_search ^
  --generations 20 ^
  --population 10 ^
  --llm http://localhost:8000 ^
  --embedding http://localhost:8002 ^
  --checkpoint 5 ^
  --verbose
goto end

:intent
echo.
echo Running Intent Classification Evolution (will reuse existing dataset if available)...
echo.
embedlab-evolve evolve ^
  --hierarchy data\sample_hierarchy.yaml ^
  --output evolution_runs\intent_classification ^
  --task-goal intent_classification ^
  --generations 20 ^
  --population 10 ^
  --llm http://localhost:8000 ^
  --embedding http://localhost:8002 ^
  --checkpoint 5 ^
  --verbose
goto end

:regenerate
echo.
echo Running Hierarchical Routing with FORCED Dataset Regeneration...
echo.
embedlab-evolve evolve ^
  --hierarchy data\sample_hierarchy.yaml ^
  --output evolution_runs\hierarchical_fresh ^
  --task-goal hierarchical_routing ^
  --regenerate ^
  --generations 20 ^
  --population 10 ^
  --llm http://localhost:8000 ^
  --embedding http://localhost:8002 ^
  --checkpoint 5 ^
  --verbose
goto end

:custom
echo.
echo Running Custom Task Goal from YAML file...
echo.
embedlab-evolve evolve ^
  --hierarchy data\sample_hierarchy.yaml ^
  --output evolution_runs\custom_goal ^
  --task-goal-file embedlab\config\custom_goals.yaml ^
  --task-goal technical_documentation_search ^
  --generations 20 ^
  --population 10 ^
  --llm http://localhost:8000 ^
  --embedding http://localhost:8002 ^
  --checkpoint 5 ^
  --verbose
goto end

:list
echo.
embedlab-evolve list-goals
echo.
pause
exit /b 0

:invalid
echo Invalid choice. Please run the script again and select 1-6.
goto end

:end
echo.
echo Evolution complete! Check the evolution_runs directory for results.
echo.
echo Dataset Reuse Info:
echo - Datasets are automatically reused if they exist and match the task goal
echo - Use --regenerate flag to force creation of new datasets
echo - Dataset metadata is saved to track task goal and generation parameters
echo.
pause
