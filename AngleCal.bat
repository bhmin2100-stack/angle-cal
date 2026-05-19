@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PYTHONW_EXE=%VENV_DIR%\Scripts\pythonw.exe"
set "READY_FILE=%VENV_DIR%\.anglecal-ready"

if not exist "%PYTHON_EXE%" (
    echo [AngleCal] First run setup: creating Python virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo [AngleCal] Python 3.9 or newer is required.
            echo Install Python from https://www.python.org/downloads/windows/
            echo During install, enable "Add python.exe to PATH".
            pause
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 goto setup_failed
)

if not exist "%READY_FILE%" (
    echo [AngleCal] Installing dependencies. This can take a few minutes on first run...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -m pip install -e .
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -c "from pathlib import Path; Path(r'%READY_FILE%').write_text('ok', encoding='utf-8')"
    if errorlevel 1 goto setup_failed
)

"%PYTHON_EXE%" -c "import angle_cal, cv2, numpy, PySide6" >nul 2>nul
if errorlevel 1 (
    echo [AngleCal] Package check failed. Reinstalling dependencies...
    del "%READY_FILE%" >nul 2>nul
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -m pip install -e .
    if errorlevel 1 goto setup_failed
    "%PYTHON_EXE%" -c "from pathlib import Path; Path(r'%READY_FILE%').write_text('ok', encoding='utf-8')"
    if errorlevel 1 goto setup_failed
)

if exist "%PYTHONW_EXE%" (
    start "" "%PYTHONW_EXE%" "%CD%\run_angle_cal.py"
) else (
    start "" "%PYTHON_EXE%" "%CD%\run_angle_cal.py"
)
exit /b 0

:setup_failed
echo.
echo [AngleCal] Setup failed. Check the messages above and run AngleCal.bat again.
pause
exit /b 1
