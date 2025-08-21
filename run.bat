@echo off
echo ============================================================
echo ASDP (AI Survey Data Processor) Application
echo Ministry of Statistics and Programme Implementation (MoSPI)
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking dependencies...

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing/updating dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo Starting the application...
echo Web interface will be available at: http://localhost:5000
echo Press Ctrl+C to stop the application
echo.

REM Run the application
python run.py

pause
