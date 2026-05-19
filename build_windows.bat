@echo off
setlocal

cd /d "%~dp0"

if not exist .venv (
    py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m PyInstaller --noconfirm --clean AngleCal.spec

echo.
echo Build complete: dist\AngleCal\AngleCal.exe
