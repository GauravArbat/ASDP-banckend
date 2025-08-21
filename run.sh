#!/bin/bash

echo "============================================================"
echo "ASDP (AI Survey Data Processor) Application"
echo "Ministry of Statistics and Programme Implementation (MoSPI)"
echo "============================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found. Checking dependencies..."

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    exit 1
fi

# Install dependencies if needed
echo "Installing/updating dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "Dependencies installed successfully!"
echo
echo "Starting the application..."
echo "Web interface will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the application"
echo

# Run the application
python3 run.py
