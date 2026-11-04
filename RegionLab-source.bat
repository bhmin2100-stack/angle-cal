@echo off
setlocal
cd /d "%~dp0"
call "%~dp0setup_angle_cal.bat"
if errorlevel 1 goto failed
if /i "%~1"=="--setup-only" exit /b 0
if /i "%~1"=="--self-test" (
    ".venv\Scripts\python.exe" "%CD%\run_region_lab.py" %*
    exit /b
)
start "" ".venv\Scripts\pythonw.exe" "%CD%\run_region_lab.py" %*
exit /b 0
:failed
echo [RegionLab source] Setup failed. Use RegionLab.bat for the verified Windows runtime.
if not defined ANGLECAL_NO_PAUSE pause
exit /b 1
