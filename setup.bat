@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo  ABI-Pipeline - Local Environment Setup (Windows)
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your PATH.
    echo Please install Python 3.10 or newer and ensure "Add Python to PATH" is checked.
    goto :error
)

:: Check if .venv already exists
if exist .venv (
    echo [INFO] Found an existing .venv folder.
    echo Upgrading dependencies inside the existing environment...
) else (
    echo [INFO] Creating virtual environment in .venv...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :error
    )
)

echo [INFO] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip. Attempting dependency installation anyway...
)

echo [INFO] Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    goto :error
)

echo.
echo ===================================================
echo  Setup Completed Successfully!
echo  You can now start the application using start.bat
echo ===================================================
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Setup failed. Please check the logs above.
echo.
pause
exit /b 1