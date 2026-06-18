@echo off
setlocal enabledelayedexpansion

:run_app
echo.
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
:: We append '< Nul' to prevent the annoying "Terminate batch job (Y/N)?" prompt on exits
.venv\Scripts\python.exe main.py < Nul

set EXIT_CODE=%errorlevel%

:: Exit code 123 is our signal to restart the app programmatically
if %EXIT_CODE% equ 123 (
    echo [INFO] Restart signal received. Reloading ABI-Pipeline...
    goto :run_app
)

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %EXIT_CODE%.
    pause
)