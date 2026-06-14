@echo off
echo Starting ABI-Pipeline via local virtual environment...
echo.

if not exist .venv (
    echo [ERROR] Virtual environment not found. 
    echo Please run setup.bat first to initialize the environment.
    echo.
    pause
    exit /b 1
)

:: Run the application directly using the venv Python executable
.venv\Scripts\python.exe main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)