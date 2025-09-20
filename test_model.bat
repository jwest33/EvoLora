@echo off
echo.
echo ================================================================
echo                 TEST TRAINED GRPO MODEL
echo ================================================================
echo.

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Get the latest adapter path
echo Looking for trained models...
echo.

REM Run the test script
python test_inference.py --mode both

pause