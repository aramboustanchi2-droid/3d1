#!/bin/bash
# Complete Installation Script for CAD 3D System (Linux/Mac)
# نصب کامل سیستم CAD 3D

echo "========================================"
echo "CAD 3D CONVERSION SYSTEM"
echo "COMPLETE INSTALLATION"
echo "========================================"
echo ""

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi
python3 --version
echo "Python OK"
echo ""

# Create/Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

echo "Activating virtual environment..."
source .venv/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install requirements
echo "========================================"
echo "Installing requirements..."
echo "========================================"
echo ""

echo "Installing from requirements_complete.txt..."
pip install -r requirements_complete.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Installation failed"
    exit 1
fi
echo ""

# Run setup
echo "========================================"
echo "Running setup script..."
echo "========================================"
echo ""

python setup_diffusion.py
if [ $? -ne 0 ]; then
    echo "WARNING: Setup had issues"
fi
echo ""

# Download models
echo "========================================"
echo "Download models?"
echo "========================================"
echo ""
echo "MiDaS depth model is required (12.5 MB)"
echo "Other models are optional and can be trained"
echo ""
read -p "Download MiDaS model now? (y/n) [default=y]: " download
download=${download:-y}

if [ "$download" = "y" ] || [ "$download" = "Y" ]; then
    python download_models.py
fi
echo ""

# Success
echo "========================================"
echo "INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run demo: python demo_diffusion.py"
echo "  2. Test ViT: python demo_vit.py"
echo "  3. Full system: python launch_neural_system.py"
echo "  4. Web server: python -m uvicorn cad3d.simple_server:app --port 8003"
echo ""
echo "Documentation:"
echo "  - INSTALL_DIFFUSION.md (installation guide)"
echo "  - DIFFUSION_MODEL_GUIDE.md (usage guide)"
echo "  - PACKAGE_INFO.json (complete info)"
echo ""
