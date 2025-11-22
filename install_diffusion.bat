@echo off
REM Quick Setup for Diffusion Model
REM راه‌اندازی سریع مدل انتشار

echo ========================================
echo DIFFUSION MODEL - QUICK SETUP
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

REM Run setup script
echo Running setup...
python setup_diffusion.py
if errorlevel 1 (
    echo ERROR: Setup failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo To use the system:
echo   1. Run: .venv\Scripts\activate
echo   2. Then: python demo_diffusion.py
echo.
pause
