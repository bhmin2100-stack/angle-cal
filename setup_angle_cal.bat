@echo off
setlocal
cd /d "%~dp0"
set "ANGLECAL_PYTHON=.venv\Scripts\python.exe"

if exist "%ANGLECAL_PYTHON%" goto check_packages
echo [AngleCal] First run setup: creating Python virtual environment...
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv ".venv"
    if errorlevel 1 exit /b 1
    goto check_packages
)
where python >nul 2>nul
if errorlevel 1 (
    echo [AngleCal] Python 3.9 or newer is required for the source BAT package.
    echo Install Python from https://www.python.org/downloads/windows/
    echo Or use the standalone RegionLab Windows ZIP; it does not need Python.
    exit /b 1
)
python -m venv ".venv"
if errorlevel 1 exit /b 1

:check_packages
rem Check real segmentation imports and calculation, even for old ready markers.
"%ANGLECAL_PYTHON%" -c "import angle_cal, cv2, numpy, PySide6; from skimage.segmentation import felzenszwalb; assert felzenszwalb(numpy.zeros((4,4,3)), min_size=1).shape == (4,4)" >nul 2>nul
if not errorlevel 1 exit /b 0

echo [AngleCal] Installing or repairing dependencies, including region segmentation...
echo [AngleCal] First setup needs Internet access. Please keep this window open.
"%ANGLECAL_PYTHON%" -m pip install -e ".[regions]"
if errorlevel 1 exit /b 1
"%ANGLECAL_PYTHON%" -c "import angle_cal, cv2, numpy, PySide6; from skimage.segmentation import felzenszwalb; assert felzenszwalb(numpy.zeros((4,4,3)), min_size=1).shape == (4,4)"
if errorlevel 1 exit /b 1
echo [AngleCal] Dependencies ready.
exit /b 0
