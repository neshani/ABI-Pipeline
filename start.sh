#!/bin/bash

echo "Starting ABI-Pipeline via local virtual environment..."
echo

if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Please run ./setup.sh first to initialize the environment."
    echo
    exit 1
fi

# Run the application directly using the venv Python executable
.venv/bin/python main.py