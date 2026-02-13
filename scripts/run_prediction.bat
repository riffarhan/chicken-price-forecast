@echo off
REM Weekly chicken price prediction pipeline.
REM Schedule with Windows Task Scheduler.

echo ============================================
echo Chicken Price Prediction Pipeline
echo Starting at %date% %time%
echo ============================================

REM Resolve project directory (parent of scripts\)
cd /d "%~dp0\.."

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python predict.py

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Pipeline completed.
) else (
    echo ERROR: Pipeline failed — check logs\ for details.
)

pause
