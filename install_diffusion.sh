#!/bin/bash
# Quick Setup for Diffusion Model (Linux/Mac)
# راه‌اندازی سریع مدل انتشار

echo "========================================"
echo "DIFFUSION MODEL - QUICK SETUP"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo ""

# Run setup script
echo "Running setup..."
python setup_diffusion.py
if [ $? -ne 0 ]; then
    echo "ERROR: Setup failed"
    exit 1
fi

echo ""
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo ""
echo "To use the system:"
echo "  1. Run: source .venv/bin/activate"
echo "  2. Then: python demo_diffusion.py"
echo ""
