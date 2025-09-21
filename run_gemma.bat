@echo off
setlocal enabledelayedexpansion

REM Gemma Evolution Runner
REM
REM Usage: run_gemma.bat [options]
REM
REM Options:
REM   quick_test          Run quick test with minimal settings (2 gen, 2 pop)
REM   pipeline_test       Run pipeline test for error checking (2 gen, 3 pop)
REM   --grpo              Use GRPO training method
REM
REM Examples:
REM   run_gemma.bat                    (standard evolution)
REM   run_gemma.bat quick_test --grpo  (quick test with GRPO)
REM   run_gemma.bat pipeline_test       (minimal pipeline test)
REM   run_gemma.bat --grpo              (GRPO training)

REM Parse command line arguments
set QUICK_TEST=0
set PIPELINE_TEST=0
set USE_GRPO=0

:parse_args
if "%1"=="" goto end_parse
if /i "%1"=="quick_test" set QUICK_TEST=1
if /i "%1"=="quick-test" set QUICK_TEST=1
if /i "%1"=="--quick-test" set QUICK_TEST=1
if /i "%1"=="pipeline_test" set PIPELINE_TEST=1
if /i "%1"=="pipeline-test" set PIPELINE_TEST=1
if /i "%1"=="--pipeline-test" set PIPELINE_TEST=1
if /i "%1"=="--grpo" set USE_GRPO=1
if /i "%1"=="grpo" set USE_GRPO=1
if /i "%1"=="--gpro" set USE_GRPO=1
if /i "%1"=="gpro" set USE_GRPO=1
shift
goto parse_args
:end_parse

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

REM Build command with all options
set CMD=python run_evolution.py evolve --config loralab/config/gemma_config.yaml

REM Add test flags (these override normal settings)
if !QUICK_TEST!==1 (
    set CMD=!CMD! --quick-test
    echo [INFO] Quick test mode enabled - 2 generations, 2 population
)
if !PIPELINE_TEST!==1 (
    set CMD=!CMD! --pipeline-test
    echo [INFO] Pipeline test mode enabled - minimal settings for error checking
)

REM Add GRPO flag if requested
if !USE_GRPO!==1 (
    set CMD=!CMD! --grpo
    echo [INFO] GRPO training mode enabled
)

REM Display what we're running
echo Running: !CMD!
echo.

REM Run the command
!CMD!

echo.
pause
