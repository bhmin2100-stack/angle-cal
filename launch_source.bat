@echo off
setlocal
cd /d "%~dp0"
if /i not "%~1"=="AngleCal" if /i not "%~1"=="RegionLab" exit /b 2
set "ANGLECAL_APP=%~1"
set "ANGLECAL_PROFILE=base"
set "ANGLECAL_SCRIPT=%~dp0run_angle_cal.py"
if /i "%ANGLECAL_APP%"=="RegionLab" (
    set "ANGLECAL_PROFILE=regions"
    set "ANGLECAL_SCRIPT=%~dp0run_region_lab.py"
)
call "%~dp0setup_angle_cal.bat" "%ANGLECAL_PROFILE%"
if errorlevel 1 goto setup_failed
if /i "%~2"=="--setup-only" exit /b 0
set "ANGLECAL_RUN_LOG=%~dp0%ANGLECAL_APP%-run.log"
echo [%ANGLECAL_APP%] Running source: %~dp0src
if /i "%~2"=="--verify" goto verify
"%~dp0.venv\Scripts\python.exe" -u "%ANGLECAL_SCRIPT%" %2 %3 %4 %5 %6 %7 %8 %9 >"%ANGLECAL_RUN_LOG%" 2>&1
goto result
:verify
if /i "%ANGLECAL_APP%"=="RegionLab" goto verify_regions
"%~dp0.venv\Scripts\python.exe" -u "%ANGLECAL_SCRIPT%" --build-info-json "%~3" >"%ANGLECAL_RUN_LOG%" 2>&1
goto result
:verify_regions
"%~dp0.venv\Scripts\python.exe" -u "%ANGLECAL_SCRIPT%" --self-test "%~3" >"%ANGLECAL_RUN_LOG%" 2>&1
:result
set "ANGLECAL_EXIT=%errorlevel%"
if "%ANGLECAL_EXIT%"=="0" exit /b 0
type "%ANGLECAL_RUN_LOG%"
echo.
echo [%ANGLECAL_APP%:START] Source execution failed with code %ANGLECAL_EXIT%.
echo Log: %ANGLECAL_RUN_LOG%
if not defined ANGLECAL_NO_PAUSE pause
exit /b %ANGLECAL_EXIT%
:setup_failed
if not defined ANGLECAL_NO_PAUSE pause
exit /b 1
