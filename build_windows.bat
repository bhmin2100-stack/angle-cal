@echo off
setlocal

cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    py -3 -m venv .venv
)

if not exist "%VENV_PYTHON%" (
    echo Failed to create the Python virtual environment.
    exit /b 1
)

"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m pip install -e .
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -m PyInstaller --noconfirm --clean AngleCal.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete: dist\AngleCal.exe
