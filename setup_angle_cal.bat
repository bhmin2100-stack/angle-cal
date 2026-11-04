@echo off
setlocal
cd /d "%~dp0"
set "ANGLECAL_PROFILE=base"
if /i "%~1"=="regions" set "ANGLECAL_PROFILE=regions"
set "ANGLECAL_PYTHON=%~dp0.venv\Scripts\python.exe"
set "ANGLECAL_SETUP_LOG=%~dp0%ANGLECAL_PROFILE%-setup.log"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "ANGLECAL_STAGE=PYTHON"
>"%ANGLECAL_SETUP_LOG%" echo Setup profile: %ANGLECAL_PROFILE%

if exist "%ANGLECAL_PYTHON%" goto check_python
echo [AngleCal] First setup: creating Python virtual environment...
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 -m venv ".venv" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if errorlevel 1 goto failed
goto check_python
:use_python
where python >nul 2>nul
if errorlevel 1 (
    >>"%ANGLECAL_SETUP_LOG%" echo Python 3.9 or newer is required for source execution.
    goto failed
)
python -m venv ".venv" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if errorlevel 1 goto failed

:check_python
"%ANGLECAL_PYTHON%" -c "import sys; print(sys.version); assert sys.version_info >= (3,9), 'Python 3.9 or newer is required'" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if errorlevel 1 goto failed
set "ANGLECAL_STAGE=BASE_DEPENDENCIES"
"%ANGLECAL_PYTHON%" -c "import cv2, numpy; from PySide6.QtWidgets import QApplication" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if not errorlevel 1 goto base_ready
echo [AngleCal] Installing missing base packages. Existing working packages are reused.
"%ANGLECAL_PYTHON%" -m pip --disable-pip-version-check install --only-binary=:all: -r "%~dp0requirements.txt" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if errorlevel 1 goto failed
"%ANGLECAL_PYTHON%" -c "import cv2, numpy; from PySide6.QtWidgets import QApplication" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if errorlevel 1 goto failed

:base_ready
if /i not "%ANGLECAL_PROFILE%"=="regions" exit /b 0
set "ANGLECAL_STAGE=REGION_DEPENDENCIES"
call :check_regions
if not errorlevel 1 exit /b 0
echo [RegionLab] Installing the optional segmentation package. Package-index access is needed once.
"%ANGLECAL_PYTHON%" -m pip --disable-pip-version-check install --only-binary=:all: -r "%~dp0requirements-regions.txt" >>"%ANGLECAL_SETUP_LOG%" 2>&1
if errorlevel 1 goto failed
call :check_regions
if errorlevel 1 goto failed
exit /b 0

:check_regions
"%ANGLECAL_PYTHON%" -c "import numpy; from skimage.segmentation import felzenszwalb; assert felzenszwalb(numpy.zeros((4,4,3)), min_size=1).shape == (4,4)" >>"%ANGLECAL_SETUP_LOG%" 2>&1
exit /b %errorlevel%

:failed
type "%ANGLECAL_SETUP_LOG%"
echo.
echo [AngleCal:%ANGLECAL_STAGE%] Setup failed. This is not a pip update request.
echo Log: %ANGLECAL_SETUP_LOG%
if /i "%ANGLECAL_STAGE%"=="REGION_DEPENDENCIES" echo AngleCal.bat can still run without the optional segmentation package.
exit /b 1
