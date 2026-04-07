@echo off
REM run_pipeline.bat
REM Script to run the complete political bias analysis pipeline on Windows

echo ========================================
echo POLITICAL BIAS ANALYSIS PIPELINE
echo ========================================

REM Create folders
echo Creating folder structure...
python create_folders.py

REM Check for config
if not exist config.json (
    echo Creating default config.json...
    echo {} > config.json
)

REM Check for .env
if not exist .env (
    echo Creating .env file...
    echo SIMULATE_MODE=true > .env
    echo LOG_LEVEL=INFO >> .env
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True)" 2>nul

REM Run pipeline
echo Starting pipeline...
if "%1"=="--real" (
    echo Using REAL API calls
    python main.py --real
) else (
    echo Using SIMULATED responses
    python main.py --simulate
)

echo ========================================
echo Pipeline complete! Check results folder.
echo ========================================
pause