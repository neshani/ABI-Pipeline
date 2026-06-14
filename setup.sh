#!/bin/bash

echo "==================================================="
echo " ABI-Pipeline - Local Environment Setup (Unix) "
echo "==================================================="
echo

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10 or newer."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ -d ".venv" ]; then
    echo "[INFO] Found an existing .venv folder. Upgrading dependencies..."
else
    echo "[INFO] Creating virtual environment in .venv..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
fi

echo "[INFO] Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip

echo "[INFO] Installing requirements..."
.venv/bin/python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Dependency installation failed."
    exit 1
fi

# Ensure startup script is executable
if [ -f "start.sh" ]; then
    chmod +x start.sh
fi

echo
echo "==================================================="
echo " Setup Completed Successfully!"
echo " You can now start the application using ./start.sh"
echo "==================================================="
echo