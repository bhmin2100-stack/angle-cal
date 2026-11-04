@echo off
setlocal

cd /d "%~dp0"

call "%~dp0setup_angle_cal.bat"
if errorlevel 1 goto setup_failed
if /i "%~1"=="--setup-only" exit /b 0

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "%CD%\run_angle_cal.py" %*
) else (
    start "" ".venv\Scripts\python.exe" "%CD%\run_angle_cal.py" %*
)
exit /b 0

:setup_failed
echo.
echo [AngleCal] Setup failed. Check the messages above and run AngleCal.bat again.
if not defined ANGLECAL_NO_PAUSE pause
exit /b 1
