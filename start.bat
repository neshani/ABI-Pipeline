@echo off
setlocal enabledelayedexpansion

:: Windows CMD has a legacy bug where if any child process exits with status 0xC000013A,
:: the batch interpreter assumes Ctrl+C was pressed and prompts "Terminate batch job (Y/N)?".
:: To bypass this, we recursively run the batch file with standard input redirected to Nul.
if "%~1"=="-FIXED_CTRL_C" (
    shift
) else (
    call <nul "%~f0" -FIXED_CTRL_C %*
    goto :EOF
)

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
.venv\Scripts\python.exe main.py

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