@echo off
setlocal
cd /d "%~dp0"
call "%~dp0setup_angle_cal.bat"
if errorlevel 1 goto failed
if /i "%~1"=="--setup-only" exit /b 0
start "" ".venv\Scripts\pythonw.exe" "%CD%\run_angle_cal.py" %*
exit /b 0
:failed
echo [AngleCal source] Setup failed. Use AngleCal.bat for the verified Windows runtime.
if not defined ANGLECAL_NO_PAUSE pause
exit /b 1
