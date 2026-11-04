@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--setup-only" goto setup
if exist "RegionLab.exe" (
    if /i "%~1"=="--self-test" (
        "RegionLab.exe" %*
        exit /b
    )
    start "" "RegionLab.exe" %*
    exit /b 0
)
if exist "dist\RegionLab\RegionLab.exe" (
    if /i "%~1"=="--self-test" (
        "dist\RegionLab\RegionLab.exe" %*
        exit /b
    )
    start "" "dist\RegionLab\RegionLab.exe" %*
    exit /b 0
)

:setup
call "%~dp0setup_angle_cal.bat"
if errorlevel 1 goto setup_failed
if /i "%~1"=="--setup-only" exit /b 0
if /i "%~1"=="--self-test" (
    ".venv\Scripts\python.exe" "%CD%\run_region_lab.py" %*
    exit /b
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "%CD%\run_region_lab.py" %*
) else (
    start "" ".venv\Scripts\python.exe" "%CD%\run_region_lab.py" %*
)
exit /b 0

:setup_failed
echo.
echo [RegionLab] Setup failed. Check the messages above and run this BAT again.
if not defined ANGLECAL_NO_PAUSE pause
exit /b 1
