@echo off
REM Complete Installation Script for CAD 3D System
REM نصب کامل سیستم CAD 3D

echo ========================================
echo CAD 3D CONVERSION SYSTEM
echo COMPLETE INSTALLATION
echo ========================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
echo Python OK
echo.

REM Create/Activate virtual environment
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo ========================================
echo Installing requirements...
echo ========================================
echo.

echo Installing from requirements_complete.txt...
pip install -r requirements_complete.txt
if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)
echo.

REM Run setup
echo ========================================
echo Running setup script...
echo ========================================
echo.

python setup_diffusion.py
if errorlevel 1 (
    echo WARNING: Setup had issues
)
echo.

REM Download models
echo ========================================
echo Download models?
echo ========================================
echo.
echo MiDaS depth model is required (12.5 MB)
echo Other models are optional and can be trained
echo.
set /p download="Download MiDaS model now? (y/n) [default=y]: "
if "%download%"=="" set download=y

if /i "%download%"=="y" (
    python download_models.py
)
echo.

REM Success
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo   1. Run demo: python demo_diffusion.py
echo   2. Test ViT: python demo_vit.py
echo   3. Full system: python launch_neural_system.py
echo   4. Web server: python -m uvicorn cad3d.simple_server:app --port 8003
echo.
echo Documentation:
echo   - INSTALL_DIFFUSION.md (installation guide)
echo   - DIFFUSION_MODEL_GUIDE.md (usage guide)
echo   - PACKAGE_INFO.json (complete info)
echo.
pause
